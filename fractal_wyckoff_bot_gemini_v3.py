import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ==========================================
# 0. CONSTANTS, LOCKS & LOGGER
# ==========================================
SYMBOL = "XAUUSD.sd"

# One magic number per strategy so trades, logs and stats can be attributed.
MAGIC_MICRO = 999999
MAGIC_SWING = 999998
SYSTEM_MAGICS = {MAGIC_MICRO, MAGIC_SWING}

BULL_SWEEP = "BULLISH_SWEEP"
BEAR_SWEEP = "BEARISH_SWEEP"

# Only used until the real offset is verified from a live tick (see BrokerClock).
BROKER_UTC_OFFSET_FALLBACK = 3

MAX_OPEN_POSITIONS = 2   # across ALL strategies (2 = one per strategy; 1 = one at a time)
MAX_TICK_AGE_SEC = 60    # refuse to trade on a stale tick

# Every mt5.* call in every thread goes through this lock.
mt5_lock = threading.RLock()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("WyckoffBot")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    _file = RotatingFileHandler(
        os.path.join(LOG_DIR, "wyckoff_bot.log"),
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    _file.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(_file)

    _cli = logging.StreamHandler()
    _cli.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(_cli)


# ==========================================
# 0.5 BROKER CLOCK (DST-safe UTC offset)
# ==========================================
class BrokerClock:
    """Broker server time is usually EET/EEST (UTC+2 winter / UTC+3 summer), so a
    hard-coded offset is wrong for ~5 months a year. This derives the offset from a
    live tick (tick.time is server time, time.time() is true UTC). A stale tick
    (market closed) isn't a whole number of hours away from UTC, so it is ignored
    and the last known offset is kept."""

    def __init__(self, fallback_hours, refresh_sec=600):
        self.offset = fallback_hours
        self.verified = False
        self._refresh_sec = refresh_sec
        self._last = 0.0

    def refresh(self, symbol):
        now = time.time()
        if now - self._last < self._refresh_sec:
            return
        self._last = now
        with mt5_lock:
            tick = mt5.symbol_info_tick(symbol)
        if tick is None or not tick.time:
            self._last = now - self._refresh_sec + 60  # retry in a minute
            return
        diff = tick.time - now
        off = round(diff / 3600)
        if abs(diff - off * 3600) > 180 or not (-12 <= off <= 14):
            self._last = now - self._refresh_sec + 60  # stale tick; retry soon
            return
        if off != self.offset or not self.verified:
            logger.info(f"Broker UTC offset {'set to' if not self.verified else 'changed to'} "
                        f"{off:+d}h (was {self.offset:+d}h, verified from live tick).")
        self.offset = off
        self.verified = True

    def tick_age(self, tick):
        """Seconds between 'now' (in broker time) and the tick, or None while the
        offset is still unverified (check is skipped rather than guessed)."""
        if not self.verified or tick is None or not tick.time:
            return None
        return time.time() + self.offset * 3600 - tick.time


broker_clock = BrokerClock(BROKER_UTC_OFFSET_FALLBACK)


# ==========================================
# 1. SHARED STATE & TELEMETRY
# ==========================================
@dataclass(frozen=True)
class Sweep:
    direction: str
    level: float
    extreme: float
    bar_time: pd.Timestamp
    close_time: pd.Timestamp
    expires_at: float
    source: str = "?"     # which strategy produced it, e.g. "ASIAN"


class TradingState:
    def __init__(self):
        # Macro (H4)
        self.macro_bias = "NEUTRAL"
        self.macro_zones = []          # valid H4 zones, most recent first
        self.macro_in_poi_h4 = False   # last CLOSED H4 bar's close is inside a zone (display only)
        self.macro_ts = 0.0

        self._sweep = None
        self.session = {"asian_high": 0.0, "asian_low": 0.0, "vol_ratio": 0.0,
                        "in_killzone": False, "vwap": 0.0, "broker_utc_offset": 0}

        self.agent_statuses = {}
        self.agent_beats = {}

        self.snapshot = {"market": {}, "account": {}, "trades": []}
        self.snapshot_ts = 0.0

        self.funnel = {}   # setup-funnel counters since start: where do setups die?

        self.state_lock = threading.Lock()

    # --- agents ---
    def update_agent_status(self, agent_name, status):
        with self.state_lock:
            self.agent_statuses[agent_name] = status

    def beat(self, agent_name, interval):
        with self.state_lock:
            self.agent_beats[agent_name] = (time.time(), interval)

    # --- macro ---
    def update_macro(self, bias, zones, in_poi_h4):
        with self.state_lock:
            self.macro_bias = bias
            self.macro_zones = [dict(z) for z in zones]
            self.macro_in_poi_h4 = bool(in_poi_h4)
            self.macro_ts = time.time()

    def get_macro(self, max_age=600):
        """(bias, has_valid_zone, fresh)"""
        with self.state_lock:
            fresh = (time.time() - self.macro_ts) <= max_age
            return self.macro_bias, bool(self.macro_zones), fresh

    def zone_touched(self, low, high, tol=0.0):
        """Live-price POI test: does the price range [low, high] of a lower-timeframe
        bar overlap any valid H4 zone? Returns the zone dict or None. This replaces
        the old 'last closed H4 close is inside the zone' flag, which could be up to
        4 hours out of date."""
        with self.state_lock:
            for z in self.macro_zones:
                if z["low"] - tol <= high and z["high"] + tol >= low:
                    return dict(z)
        return None

    # --- session ---
    def update_session_telemetry(self, asian_high, asian_low, vol_ratio, in_kz, vwap, offset):
        with self.state_lock:
            self.session = {"asian_high": float(asian_high), "asian_low": float(asian_low),
                            "vol_ratio": float(vol_ratio), "in_killzone": bool(in_kz),
                            "vwap": float(vwap), "broker_utc_offset": int(offset)}

    # --- sweep ---
    def set_sweep(self, sweep):
        with self.state_lock:
            self._sweep = sweep

    def get_sweep(self):
        with self.state_lock:
            if self._sweep is not None and time.time() > self._sweep.expires_at:
                self._sweep = None
            return self._sweep

    def clear_sweep(self):
        with self.state_lock:
            self._sweep = None

    # --- setup funnel ---
    def count(self, key, n=1):
        with self.state_lock:
            self.funnel[key] = self.funnel.get(key, 0) + n

    # --- MT5 snapshot ---
    def update_snapshot(self, market, account, trades):
        with self.state_lock:
            self.snapshot = {"market": market, "account": account, "trades": trades}
            self.snapshot_ts = time.time()

    # --- everything the dashboard needs ---
    def export(self):
        now = time.time()
        with self.state_lock:
            if self._sweep is not None and now > self._sweep.expires_at:
                self._sweep = None
            sw = self._sweep
            sweep = None if sw is None else {
                "direction": sw.direction, "level": sw.level, "extreme": sw.extreme,
                "source": sw.source, "expires_in_sec": max(0, round(sw.expires_at - now)),
            }

            agents = {}
            for name, status in self.agent_statuses.items():
                ts, interval = self.agent_beats.get(name, (0.0, 0))
                age = (now - ts) if ts else None
                agents[name] = {
                    "status": status,
                    "age": None if age is None else round(age),
                    "stale": age is None or age > max(60, interval * 10),
                }

            return {
                "macro": {
                    "bias": self.macro_bias,
                    "in_poi_h4": self.macro_in_poi_h4,
                    "zones": [dict(z) for z in self.macro_zones],
                    "fresh": (now - self.macro_ts) <= 600,
                },
                "session": dict(self.session),
                "sweep": sweep,
                "agents": agents,
                "market": self.snapshot["market"],
                "account": self.snapshot["account"],
                "trades": self.snapshot["trades"],
                "snapshot_age": round(now - self.snapshot_ts) if self.snapshot_ts else None,
                "funnel": dict(self.funnel),
            }


shared_state = TradingState()


# ==========================================
# 1.5 SHARED PATTERN HELPERS
# ==========================================
def volume_ratio(df, j, n=20):
    """tick_volume of bar j relative to the mean of the n bars before it."""
    if j < n:
        return 0.0
    ma = df['tick_volume'].iloc[j - n:j].mean()
    return float(df['tick_volume'].iloc[j] / ma) if ma > 0 else 0.0


def detect_sweep(df, j, bullish, swing_n=30, gap=3, vol_mult=1.5):
    """Spring (bullish) / upthrust (bearish) on bar j (positional) of df:
    pierces the prior swing extreme, closes back inside, candle direction
    confirms, climax volume. Returns (hit, level, extreme)."""
    if j < swing_n + gap or j < 20:
        return False, 0.0, 0.0
    bar = df.iloc[j]
    climax = volume_ratio(df, j) > vol_mult
    if bullish:
        level = float(df['low'].iloc[j - gap - swing_n:j - gap].min())
        hit = (bar['low'] < level and bar['close'] > level
               and bar['close'] > bar['open'] and climax)
        return bool(hit), level, float(bar['low'])
    level = float(df['high'].iloc[j - gap - swing_n:j - gap].max())
    hit = (bar['high'] > level and bar['close'] < level
           and bar['close'] < bar['open'] and climax)
    return bool(hit), level, float(bar['high'])


# ==========================================
# 2. BASE AGENT
# ==========================================
class BaseAgent(threading.Thread):
    def __init__(self, name, symbol, timeframe, sleep_interval):
        super().__init__(name=name, daemon=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.sleep_interval = sleep_interval
        self._stop_event = threading.Event()
        self.last_beat = time.time()
        shared_state.update_agent_status(name, "Initializing...")

    @property
    def running(self):
        return not self._stop_event.is_set()

    def fetch_data(self, bars=100, closed_only=True):
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if closed_only:
            df = df.iloc[:-1]
        return df.reset_index(drop=True)

    def step(self):
        raise NotImplementedError

    def run(self):
        logger.info(f"[{self.name}] Started.")
        while not self._stop_event.is_set():
            self.last_beat = time.time()
            shared_state.beat(self.name, self.sleep_interval)
            try:
                self.step()
            except Exception as e:
                logger.error(f"[{self.name}] Error in step(): {e}", exc_info=True)
            self._stop_event.wait(self.sleep_interval)

    def stop(self):
        self._stop_event.set()


# ==========================================
# 3. MACRO AGENT (H4) - BIAS & ZONES
# ==========================================
class MacroAgent(BaseAgent):
    EXPANSION_MULT = 1.5
    ATR_PERIOD = 14
    MAX_ZONE_AGE = 100   # H4 bars
    MAX_ZONES = 3

    def step(self):
        df = self.fetch_data(bars=300)
        if df is None or len(df) < 210:
            shared_state.update_agent_status(self.name, "Awaiting Data...")
            return

        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        ema200 = df['close'].ewm(span=200, adjust=False).mean()
        bias = "BULLISH" if ema50.iloc[-1] > ema200.iloc[-1] else "BEARISH"
        bullish = bias == "BULLISH"

        prev_close = df['close'].shift(1)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()),
        )
        atr = tr.rolling(self.ATR_PERIOD).mean().shift(1)
        big = tr > self.EXPANSION_MULT * atr

        direction = (df['close'] > df['open']) if bullish else (df['close'] < df['open'])
        idx = np.flatnonzero((direction & big).to_numpy())
        idx = idx[idx >= 1]

        # Walk expansions newest -> oldest and keep every zone that has not been
        # closed through (previously only the newest expansion was considered, so
        # one invalidated zone hid older valid ones).
        n = len(df)
        zones = []
        for i in idx[::-1]:
            i = int(i)
            if (n - 1) - i > self.MAX_ZONE_AGE:
                break
            z_high = float(df['high'].iloc[i - 1])
            z_low = float(df['low'].iloc[i - 1])
            after = df['close'].iloc[i + 1:]
            invalid = (after < z_low).any() if bullish else (after > z_high).any()
            if not invalid:
                zones.append({"low": z_low, "high": z_high, "age": (n - 1) - i})
                if len(zones) >= self.MAX_ZONES:
                    break

        price = float(df['close'].iloc[-1])
        in_poi_h4 = any(z["low"] <= price <= z["high"] for z in zones)

        shared_state.update_macro(bias, zones, in_poi_h4)
        shared_state.update_agent_status(
            self.name, f"Bias: {bias} | {len(zones)} valid zone(s) | H4 close in POI: {in_poi_h4}")


# ==========================================
# 4. KILLZONE AGENT (M15) - ASIAN-RANGE SWEEPS
# ==========================================
class KillzoneAgent(BaseAgent):
    ASIAN_END_HOUR = 6            # Asian session = 00:00-06:00 UTC
    MIN_ASIAN_BARS = 20           # of 24 M15 bars; otherwise the range is not trusted
    KILLZONES = ((7, 10), (12, 15))   # inclusive UTC hours (London, New York)

    def __init__(self, name, symbol, timeframe, sleep_interval, clock,
                 allow_shorts=True, sweep_ttl_sec=3600):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.clock = clock
        self.allow_shorts = allow_shorts
        self.sweep_ttl_sec = sweep_ttl_sec
        self.last_bar_time = None
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.asian_date = None

    def step(self):
        self.clock.refresh(self.symbol)
        offset = self.clock.offset

        bias, has_zone, fresh = shared_state.get_macro()
        df = self.fetch_data(bars=130)   # ~32h of M15, covers today's Asian session
        if df is None or len(df) < 60:
            return

        n = len(df)
        bar = df.iloc[-1]

        t_utc = df['time'] - pd.Timedelta(hours=offset)
        hour = t_utc.dt.hour
        date = t_utc.dt.date

        # Daily anchored VWAP (telemetry only; not used in any entry rule yet)
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['tick_volume']
        cum_v = vol.groupby(date).cumsum()
        cum_tpv = (tp * vol).groupby(date).cumsum()
        vwap = float((cum_tpv / np.maximum(cum_v, 1)).iloc[-1])

        # Asian range for TODAY only. Never reuse yesterday's range if today's
        # session is missing/incomplete.
        today = date.iloc[-1]
        asian = df[(date == today) & (hour < self.ASIAN_END_HOUR)]
        if len(asian) >= self.MIN_ASIAN_BARS:
            self.asian_high = float(asian['high'].max())
            self.asian_low = float(asian['low'].min())
            self.asian_date = today
        elif self.asian_date != today:
            self.asian_high = self.asian_low = 0.0

        hour_now = int(hour.iloc[-1])
        in_kz = any(a <= hour_now <= b for a, b in self.KILLZONES)
        vol_ratio = volume_ratio(df, n - 1)

        shared_state.update_session_telemetry(
            self.asian_high, self.asian_low, vol_ratio, in_kz, vwap, offset)

        if bar['time'] == self.last_bar_time:
            return
        self.last_bar_time = bar['time']

        # ---- gating ----
        if bias not in ("BULLISH", "BEARISH") or not fresh:
            shared_state.update_agent_status(self.name, "Idle (Macro data stale)")
            return
        if not has_zone:
            shared_state.update_agent_status(self.name, "Idle (No valid H4 zone)")
            return
        if bias == "BEARISH" and not self.allow_shorts:
            shared_state.update_agent_status(self.name, "Idle (Shorts Disabled)")
            return
        if not in_kz:
            shared_state.update_agent_status(
                self.name, f"Outside Killzone (Asian {self.asian_low:.1f}-{self.asian_high:.1f})")
            return
        if self.asian_high <= 0 or self.asian_low <= 0:
            shared_state.update_agent_status(self.name, "Idle (No valid Asian range today)")
            return
        if shared_state.get_sweep() is not None:
            shared_state.update_agent_status(self.name, "Sweep Latched. Waiting for execution.")
            return

        # ---- Asian-range sweep on the last closed M15 bar ----
        shared_state.count("kz_bars_scanned_in_window")
        bullish = bias == "BULLISH"
        climax = vol_ratio > 1.5

        prior = df.iloc[:-1]
        prior_after = prior[(date.iloc[:-1] == today) & (hour.iloc[:-1] >= self.ASIAN_END_HOUR)]

        if bullish:
            level, extreme = self.asian_low, float(bar['low'])
            pattern = (bar['low'] < level and bar['close'] > level
                       and bar['close'] > bar['open'] and climax)
            already_taken = bool((prior_after['low'] < level).any())
            direction = BULL_SWEEP
        else:
            level, extreme = self.asian_high, float(bar['high'])
            pattern = (bar['high'] > level and bar['close'] < level
                       and bar['close'] < bar['open'] and climax)
            already_taken = bool((prior_after['high'] > level).any())
            direction = BEAR_SWEEP

        if not pattern:
            shared_state.update_agent_status(
                self.name, f"Scanning KZ | Vol {vol_ratio:.2f}x | Asian {self.asian_low:.1f}-{self.asian_high:.1f}")
            return

        shared_state.count("kz_sweep_pattern")
        if already_taken:
            logger.info(f"[{self.name}] Asian-level sweep ignored: level was already taken earlier "
                        f"today (not fresh liquidity).")
            shared_state.update_agent_status(self.name, "Sweep ignored (level already taken today)")
            return

        zone = shared_state.zone_touched(float(bar['low']), float(bar['high']))
        if zone is None:
            logger.info(f"[{self.name}] Asian sweep ignored: sweep bar did not touch a valid H4 zone.")
            shared_state.update_agent_status(self.name, "Sweep ignored (outside H4 zones)")
            return

        shared_state.count("kz_fresh_and_at_h4_zone")
        bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
        shared_state.set_sweep(Sweep(
            direction=direction, level=level, extreme=extreme,
            bar_time=bar['time'], close_time=bar['time'] + bar_delta,
            expires_at=time.time() + self.sweep_ttl_sec, source="ASIAN",
        ))
        logger.info(f"[{self.name}] {direction} latched at {bar['time']} "
                    f"(Asian level {level:.2f}, extreme {extreme:.2f}, "
                    f"H4 zone {zone['low']:.2f}-{zone['high']:.2f})")
        shared_state.update_agent_status(self.name, f"LATCHED: {direction} ({level:.2f})")


# ==========================================
# 5. EXECUTION BASE + SWING (H1) + MICRO (M1)
# ==========================================
class ExecutionAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, magic,
                 use_dynamic_lot, fixed_lot, risk_pct,
                 rr=2.0, deviation=20, cooldown_sec=300,
                 min_sl_spread_mult=3.0, max_spread_points=None):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.magic = magic
        self.use_dynamic_lot = use_dynamic_lot
        self.fixed_lot = fixed_lot
        self.risk_pct = risk_pct
        self.rr = rr
        self.deviation = deviation
        self.cooldown_sec = cooldown_sec
        self.min_sl_spread_mult = min_sl_spread_mult
        self.max_spread_points = max_spread_points
        self.cooldown_until = 0.0
        self.last_bar_time = None

    def has_open_position(self):
        """True if this strategy already has a position, or the system is at
        MAX_OPEN_POSITIONS. Fails closed on error. Call with mt5_lock held."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return True
        mine = [p for p in positions if p.magic in SYSTEM_MAGICS]
        if any(p.magic == self.magic for p in mine):
            return True
        return len(mine) >= MAX_OPEN_POSITIONS

    @staticmethod
    def _filling_mode(info):
        if info.filling_mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _skip(self, reason):
        logger.warning(f"[{self.name}] Trade skipped: {reason}")
        return False

    def calculate_lot_size(self, info, entry, sl):
        step = info.volume_step
        vmin, vmax = info.volume_min, info.volume_max

        if not self.use_dynamic_lot:
            lot = math.floor(self.fixed_lot / step + 1e-9) * step
            lot = max(vmin, min(lot, vmax))
            return round(round(lot / step) * step, 8)

        acct = mt5.account_info()
        if acct is None:
            logger.warning(f"[{self.name}] No account info.")
            return None

        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value_loss or info.trade_tick_value
        if tick_size <= 0 or tick_value <= 0:
            logger.warning(f"[{self.name}] Bad tick size/value.")
            return None

        risk_per_lot = (abs(entry - sl) / tick_size) * tick_value
        if risk_per_lot <= 0:
            return None

        raw_lot = (acct.balance * (self.risk_pct / 100.0)) / risk_per_lot
        lot = math.floor(raw_lot / step + 1e-9) * step

        if lot < vmin:
            logger.warning(f"[{self.name}] Risk-based lot {raw_lot:.4f} < min {vmin} "
                           f"(SL distance {abs(entry - sl):.2f}); skipping to stay within "
                           f"{self.risk_pct}% risk.")
            return None

        lot = min(lot, vmax)
        return round(round(lot / step) * step, 8)

    def execute_trade(self, direction, sl_raw, tag=None):
        """Returns True only if an order was actually filled."""
        is_buy = direction == BULL_SWEEP

        with mt5_lock:
            if self.has_open_position():
                return False

            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
            if info is None or tick is None:
                return self._skip(f"no symbol/tick info ({mt5.last_error()})")

            terminal = mt5.terminal_info()
            if terminal is None or not terminal.connected:
                return self._skip("MT5 terminal is not connected to the broker")

            age = broker_clock.tick_age(tick)
            if age is not None and abs(age) > MAX_TICK_AGE_SEC:
                return self._skip(f"stale tick ({age:.0f}s off server time)")

            spread = tick.ask - tick.bid
            if (self.max_spread_points is not None
                    and spread / info.point > self.max_spread_points):
                return self._skip(f"spread too wide ({spread / info.point:.0f} pts)")

            entry = tick.ask if is_buy else tick.bid
            sl = sl_raw - spread if is_buy else sl_raw + spread

            if (is_buy and sl >= entry) or (not is_buy and sl <= entry):
                return self._skip("SL on wrong side of entry")

            risk = abs(entry - sl)
            min_dist = max(info.trade_stops_level * info.point,
                           spread * self.min_sl_spread_mult)
            if risk < min_dist:
                return self._skip(f"SL distance {risk:.2f} < min {min_dist:.2f}")

            tp = entry + risk * self.rr if is_buy else entry - risk * self.rr

            lot = self.calculate_lot_size(info, entry, sl)
            if lot is None:
                return False

            digits = info.digits
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": round(entry, digits),
                "sl": round(sl, digits),
                "tp": round(tp, digits),
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": (tag or self.name)[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(info),
            }
            result = mt5.order_send(request)

        if result is None:
            logger.warning(f"[{self.name}] order_send returned None: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"[{self.name}] Trade rejected: retcode={result.retcode} {result.comment}")
            return False

        logger.info(f"[{self.name}] Executed {'BUY' if is_buy else 'SELL'} {lot} lots at "
                    f"{result.price}. SL: {request['sl']}, TP: {request['tp']} [{request['comment']}]")
        return True


class SwingAgent(ExecutionAgent):
    """H1 entries: liquidity sweep at an H4 zone, then an FVG in the bias direction.

    Sequence required (all on CLOSED H1 bars):
      1. spring/upthrust within the last `sweep_lookback` bars whose range touched
         a valid H4 zone (live price test, not the lagging H4-close flag),
      2. the sweep is not invalidated (no close beyond its extreme since),
      3. FVG (c1, c2, c3) where the displacement candle c2 forms AFTER the sweep bar.
    """

    def __init__(self, *args, allow_shorts=False, sweep_lookback=6, swing_n=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.sweep_lookback = sweep_lookback
        self.swing_n = swing_n

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(
                self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        bias, has_zone, fresh = shared_state.get_macro()
        if not (fresh and has_zone) or bias not in ("BULLISH", "BEARISH"):
            shared_state.update_agent_status(self.name, "Idle (No valid H4 zone)")
            return
        if bias == "BEARISH" and not self.allow_shorts:
            shared_state.update_agent_status(self.name, "Idle (Shorts Disabled)")
            return

        with mt5_lock:
            busy = self.has_open_position()
        if busy:
            shared_state.update_agent_status(self.name, "Idle (Position open)")
            return

        df = self.fetch_data(bars=90)
        if df is None or len(df) < 60:
            return

        n = len(df)
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c3['time'] == self.last_bar_time:
            return
        self.last_bar_time = c3['time']

        bullish = bias == "BULLISH"
        shared_state.update_agent_status(self.name, f"Hunting H1 sweep + FVG ({bias})")

        if bullish:
            has_gap = c1['high'] < c3['low']
            is_disp = c2['close'] > c2['open']
            is_hold = c3['close'] > c1['high']
            direction, sl_raw = BULL_SWEEP, float(c1['low'])
        else:
            has_gap = c1['low'] > c3['high']
            is_disp = c2['close'] < c2['open']
            is_hold = c3['close'] < c1['low']
            direction, sl_raw = BEAR_SWEEP, float(c1['high'])

        if not (has_gap and is_disp and is_hold):
            return
        shared_state.count("swing_h1_fvg")

        # Find the most recent valid sweep at/before c1 (so c2 forms after it)
        found = None
        for j in range(n - 3, n - 3 - self.sweep_lookback - 1, -1):
            hit, level, extreme = detect_sweep(df, j, bullish, swing_n=self.swing_n)
            if not hit:
                continue
            bar_j = df.iloc[j]
            zone = shared_state.zone_touched(float(bar_j['low']), float(bar_j['high']))
            if zone is None:
                continue
            after = df['close'].iloc[j + 1:]
            failed = (after < extreme).any() if bullish else (after > extreme).any()
            if failed:
                continue
            found = (j, level, extreme, zone)
            break

        if found is None:
            logger.info(f"[{self.name}] H1 FVG ({bias}) found but no valid sweep at an H4 zone "
                        f"in the last {self.sweep_lookback} bars; skipped.")
            return

        shared_state.count("swing_h1_fvg_with_sweep")
        j, level, extreme, zone = found
        logger.info(f"[{self.name}] H1 sweep (bar {df['time'].iloc[j]}, level {level:.2f}, "
                    f"extreme {extreme:.2f}) at H4 zone {zone['low']:.2f}-{zone['high']:.2f} "
                    f"+ FVG confirmed ({bias}).")
        if self.execute_trade(direction, sl_raw, tag=f"{self.name}:H1SWEEP"):
            shared_state.count("swing_h1_trades")
            self.cooldown_until = time.time() + self.cooldown_sec


class MicroAgent(ExecutionAgent):
    """M1 FVG entries after a latched M15 sweep."""

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(
                self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        sweep = shared_state.get_sweep()
        if sweep is None:
            shared_state.update_agent_status(self.name, "Idle (Waiting for Sweep)")
            return

        with mt5_lock:
            busy = self.has_open_position()
        if busy:
            shared_state.update_agent_status(self.name, "Idle (Position open)")
            return

        shared_state.update_agent_status(
            self.name, f"Hunting M1 FVG ({sweep.direction}, {sweep.source})")

        df = self.fetch_data(bars=10)
        if df is None or len(df) < 5:
            return

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c3['time'] == self.last_bar_time:
            return
        self.last_bar_time = c3['time']

        bullish = sweep.direction == BULL_SWEEP
        if (bullish and c3['close'] < sweep.extreme) or (not bullish and c3['close'] > sweep.extreme):
            logger.info(f"[{self.name}] Sweep failed (close beyond sweep extreme); cleared.")
            shared_state.clear_sweep()
            return

        if c2['time'] < sweep.close_time:
            return

        if bullish:
            has_gap = c1['high'] < c3['low']
            is_disp = c2['close'] > c2['open']
            is_hold = c3['close'] > c1['high']
            sl_raw = float(c1['low'])
        else:
            has_gap = c1['low'] > c3['high']
            is_disp = c2['close'] < c2['open']
            is_hold = c3['close'] < c1['low']
            sl_raw = float(c1['high'])

        if has_gap and is_disp and is_hold:
            logger.info(f"[{self.name}] M1 FVG confirmed ({sweep.direction}, {sweep.source}).")
            shared_state.count("micro_fvg_after_sweep")
            if self.execute_trade(sweep.direction, sl_raw, tag=f"{self.name}:{sweep.source}"):
                shared_state.count("micro_trades")
                shared_state.clear_sweep()
                self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================
# 6. TRADE MANAGER AGENT
# ==========================================
class TradeManagerAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval,
                 rr=2.0, be_trigger_r=1.0, be_buffer_points=20):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.rr = rr
        self.be_trigger_r = be_trigger_r
        self.be_buffer_points = be_buffer_points
        self._risks = {}

    def step(self):
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            info = mt5.symbol_info(self.symbol)
        if positions is None or info is None:
            return

        mine = [p for p in positions if p.magic in SYSTEM_MAGICS]
        live = {p.ticket for p in mine}
        shared_state.update_agent_status(
            self.name, f"Monitoring {len(mine)} open position(s) for break-even")

        point, digits = info.point, info.digits
        stops_dist = info.trade_stops_level * point
        buffer = self.be_buffer_points * point

        for pos in mine:
            if pos.sl == 0.0 or pos.tp == 0.0:
                continue

            is_buy = pos.type == mt5.POSITION_TYPE_BUY

            if pos.ticket not in self._risks:
                on_loss_side = (pos.sl < pos.price_open) if is_buy else (pos.sl > pos.price_open)
                self._risks[pos.ticket] = (abs(pos.price_open - pos.sl) if on_loss_side
                                           else abs(pos.tp - pos.price_open) / self.rr)

            risk = self._risks[pos.ticket]
            profit = (pos.price_current - pos.price_open) if is_buy \
                else (pos.price_open - pos.price_current)
            secured = (pos.sl >= pos.price_open) if is_buy else (pos.sl <= pos.price_open)

            if secured or profit < risk * self.be_trigger_r:
                continue

            new_sl = round(pos.price_open + buffer, digits) if is_buy \
                else round(pos.price_open - buffer, digits)

            if (is_buy and new_sl >= pos.price_current) or (not is_buy and new_sl <= pos.price_current):
                continue
            if abs(pos.price_current - new_sl) < stops_dist:
                continue

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
                "magic": pos.magic,
            }
            with mt5_lock:
                result = mt5.order_send(request)

            if result is None:
                logger.warning(f"[{self.name}] BE modify returned None: {mt5.last_error()}")
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.warning(f"[{self.name}] BE modify failed: retcode={result.retcode} {result.comment}")
            else:
                logger.info(f"[{self.name}] Ticket {pos.ticket} secured at break-even ({new_sl}).")

        self._risks = {t: v for t, v in self._risks.items() if t in live}


# ==========================================
# 7. TELEMETRY AGENT (the web layer never touches MT5)
# ==========================================
class TelemetryAgent(BaseAgent):
    def step(self):
        with mt5_lock:
            acct = mt5.account_info()
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
            positions = mt5.positions_get(symbol=self.symbol)

        market = {}
        if info is not None and tick is not None:
            market = {"symbol": info.name, "bid": tick.bid, "ask": tick.ask,
                      "spread": round((tick.ask - tick.bid) / info.point, 1)}

        account = {}
        if acct is not None:
            account = {"balance": acct.balance, "equity": acct.equity,
                       "profit": acct.profit, "margin_free": acct.margin_free}

        trades = []
        for p in (positions or []):
            if p.magic in SYSTEM_MAGICS:
                trades.append({
                    "ticket": p.ticket, "agent": p.comment,
                    "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": p.volume, "open": p.price_open, "current": p.price_current,
                    "sl": p.sl, "tp": p.tp, "profit": p.profit,
                })

        shared_state.update_snapshot(market, account, trades)
        shared_state.update_agent_status(self.name, "Snapshotting MT5 state")


# ==========================================
# 8. WEB DASHBOARD
# ==========================================
app = FastAPI(title="Wyckoff Bot Terminal")


@app.get("/api/state")
def get_system_state():
    return shared_state.export()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XAUUSD Multi-Agent Command Terminal</title>
<style>
:root{--bg:#0c0d0e;--card:#16181b;--line:#262a2f;--fg:#e1e3e6;--mut:#8b9098;--bull:#3fb950;--bear:#f85149;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;padding:20px;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.row{display:flex;justify-content:space-between;gap:12px;padding:3px 0;font-size:14px}
.row span:first-child{color:var(--mut)}
.bull{color:var(--bull)}.bear{color:var(--bear)}.warn{color:var(--warn)}
.wide{grid-column:1/-1;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:500}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.badge{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}
.on{background:#238636;color:#fff}.off{background:#30363d;color:#8b949e}
.empty{color:var(--mut);font-size:13px}
</style></head><body>
<h1>XAUUSD Multi-Agent Command Terminal</h1>
<div class="sub" id="conn">connecting...</div>
<div class="grid">
  <div class="card"><h2>Market</h2><div id="market"></div></div>
  <div class="card"><h2>Account</h2><div id="account"></div></div>
  <div class="card"><h2>H4 Macro</h2><div id="macro"></div></div>
  <div class="card"><h2>Killzones &amp; Liquidity (M15)</h2><div id="session"></div></div>
  <div class="card"><h2>Setup funnel (since start)</h2><div id="funnel"></div></div>
  <div class="card wide"><h2>Agents</h2><div id="agents"></div></div>
  <div class="card wide"><h2>Open positions</h2><div id="trades"></div></div>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=2) => (v === null || v === undefined || isNaN(v)) ? '--' : Number(v).toFixed(d);
const row = (k, v, cls='') => `<div class="row"><span>${esc(k)}</span><span class="${cls}">${v}</span></div>`;
const pnl = v => (v === null || v === undefined) ? '--' : `<span class="${v >= 0 ? 'bull' : 'bear'}">${num(v)}</span>`;

function render(d) {
  const m = d.market || {}, a = d.account || {}, mc = d.macro || {}, s = d.session || {};

  $('market').innerHTML =
    row('Symbol', esc(m.symbol || '--')) +
    row('Bid / Ask', `${num(m.bid)} / ${num(m.ask)}`) +
    row('Spread', `${num(m.spread, 1)} pts`);

  $('account').innerHTML =
    row('Balance', '$' + num(a.balance)) +
    row('Equity', '$' + num(a.equity)) +
    row('Free margin', '$' + num(a.margin_free)) +
    row('Floating PnL', pnl(a.profit));

  const bcls = mc.bias === 'BULLISH' ? 'bull' : (mc.bias === 'BEARISH' ? 'bear' : '');
  const zones = mc.zones || [];
  $('macro').innerHTML =
    row('Bias', esc(mc.bias || '--'), bcls) +
    row('H4 close in POI', mc.in_poi_h4 ? 'yes' : 'no') +
    row('Macro data', mc.fresh ? 'fresh' : 'STALE', mc.fresh ? '' : 'bear') +
    (zones.length
      ? zones.map((z, i) => row('Zone ' + (i + 1), `${num(z.low)} - ${num(z.high)} (${z.age} bars)`)).join('')
      : row('Zones', 'none valid', 'warn'));

  const sw = d.sweep;
  $('session').innerHTML =
    row('Killzone', s.in_killzone ? '<span class="badge on">OPEN</span>' : '<span class="badge off">CLOSED</span>') +
    row('Asian range', s.asian_high ? `${num(s.asian_low)} - ${num(s.asian_high)}` : 'none today') +
    row('Daily VWAP', s.vwap ? num(s.vwap) : '--') +
    row('Volume ratio', num(s.vol_ratio) + 'x') +
    row('Broker UTC offset', (s.broker_utc_offset >= 0 ? '+' : '') + s.broker_utc_offset + 'h') +
    row('Latched sweep', sw ? esc(sw.direction) + ' [' + esc(sw.source) + ']' : 'none', sw ? 'warn' : '') +
    (sw ? row('Level / extreme', `${num(sw.level)} / ${num(sw.extreme)}`) +
          row('Expires in', sw.expires_in_sec + 's') : '');

  const f = d.funnel || {};
  const fk = Object.keys(f).sort();
  $('funnel').innerHTML = fk.length ? fk.map(k => row(k.replace(/_/g, ' '), f[k])).join('')
                                    : '<div class="empty">no setups seen yet</div>';

  const names = Object.keys(d.agents || {});
  $('agents').innerHTML = '<table><tr><th></th><th>Agent</th><th>Status</th><th>Last loop</th></tr>' +
    names.map(n => {
      const g = d.agents[n];
      const col = g.stale ? 'var(--bear)' : 'var(--bull)';
      return `<tr><td><span class="dot" style="background:${col}"></span></td><td>${esc(n)}</td>` +
             `<td>${esc(g.status)}</td><td>${g.age === null ? '--' : g.age + 's ago'}</td></tr>`;
    }).join('') + '</table>';

  const t = d.trades || [];
  $('trades').innerHTML = t.length
    ? '<table><tr><th>Ticket</th><th>Tag</th><th>Side</th><th>Lots</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>PnL</th></tr>' +
      t.map(x => `<tr><td>${esc(x.ticket)}</td><td>${esc(x.agent)}</td>` +
        `<td class="${x.side === 'BUY' ? 'bull' : 'bear'}">${esc(x.side)}</td>` +
        `<td>${num(x.volume)}</td><td>${num(x.open)}</td><td>${num(x.current)}</td>` +
        `<td>${num(x.sl)}</td><td>${num(x.tp)}</td><td>${pnl(x.profit)}</td></tr>`).join('') + '</table>'
    : '<div class="empty">No open positions.</div>';

  const age = d.snapshot_age;
  $('conn').innerHTML = 'live · ' + new Date().toLocaleTimeString() +
    (age !== null && age > 15 ? ' · <span class="bear">MT5 snapshot is ' + age + 's old</span>' : '');
}

async function tick() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    if (!r.ok) throw new Error(r.status);
    render(await r.json());
  } catch (e) {
    $('conn').innerHTML = '<span class="bear">disconnected from bot</span>';
  }
}
tick();
setInterval(tick, 2000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return DASHBOARD_HTML


def run_web_server():
    try:
        # Localhost only: the API exposes balance/equity. Add auth before binding wider.
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    except Exception:
        logger.error("Web server failed to start/run", exc_info=True)


# ==========================================
# 9. ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        quit()

    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Could not select {SYMBOL}: {mt5.last_error()}")
        mt5.shutdown()
        quit()

    RR = 2.0
    ALLOW_SHORTS = True

    clock = broker_clock

    macro = MacroAgent("Macro_H4", SYMBOL, mt5.TIMEFRAME_H4, sleep_interval=60)
    killzone = KillzoneAgent("Killzone_M15", SYMBOL, mt5.TIMEFRAME_M15, sleep_interval=15,
                             clock=clock, allow_shorts=ALLOW_SHORTS, sweep_ttl_sec=3600)
    swing = SwingAgent("Swing_H1", SYMBOL, mt5.TIMEFRAME_H1, sleep_interval=15,
                       magic=MAGIC_SWING, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=1.0,
                       rr=RR, deviation=20, cooldown_sec=3600, min_sl_spread_mult=3.0,
                       max_spread_points=None, allow_shorts=ALLOW_SHORTS,
                       sweep_lookback=6, swing_n=30)
    micro = MicroAgent("Micro_M1", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2,
                       magic=MAGIC_MICRO, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=1.0,
                       rr=RR, deviation=20, cooldown_sec=300, min_sl_spread_mult=3.0,
                       max_spread_points=None)
    manager = TradeManagerAgent("Manager", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=3,
                                rr=RR, be_trigger_r=1.0, be_buffer_points=20)
    telemetry = TelemetryAgent("Telemetry", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2)

    agents = [macro, killzone, swing, micro, manager, telemetry]
    for a in agents:
        a.start()

    threading.Thread(target=run_web_server, name="WebServer", daemon=True).start()

    warned = set()
    terminal_was_ok = True
    try:
        logger.info("System running. Dashboard: http://127.0.0.1:8000  (Ctrl+C to stop)")
        while True:
            time.sleep(5)

            # MT5 terminal heartbeat: None = terminal/API gone (fatal);
            # connected=False = terminal is up but the broker link is down (warn, trades are blocked).
            with mt5_lock:
                terminal = mt5.terminal_info()
            if terminal is None:
                logger.critical("!! FATAL: MT5 terminal closed or API unreachable. Shutting down. !!")
                break
            if not terminal.connected and terminal_was_ok:
                logger.error("!! MT5 terminal lost its broker connection: new trades are blocked. !!")
            elif terminal.connected and not terminal_was_ok:
                logger.warning("MT5 broker connection restored.")
            terminal_was_ok = terminal.connected

            for a in agents:
                if a.name in warned:
                    continue
                if not a.is_alive():
                    logger.error(f"!! [{a.name}] thread is DEAD.")
                    warned.add(a.name)
                elif time.time() - a.last_beat > max(60, a.sleep_interval * 10):
                    logger.warning(f"!! [{a.name}] has not completed a loop in "
                                   f"{time.time() - a.last_beat:.0f}s (stuck MT5 call?).")
                    warned.add(a.name)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down cleanly...")
        for a in agents:
            a.stop()
        for a in agents:
            a.join(timeout=10)
        mt5.shutdown()
        logger.info("Shutdown complete.")