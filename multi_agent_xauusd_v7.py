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

MAGIC_SWING = 999998
MAGIC_LEVEL = 999997
MAGIC_PYRAMID = 999996
SYSTEM_MAGICS = {MAGIC_SWING, MAGIC_LEVEL, MAGIC_PYRAMID}

BULL_SWEEP = "BULLISH_SWEEP"
BEAR_SWEEP = "BEARISH_SWEEP"

BROKER_UTC_OFFSET_FALLBACK = 3

# Documented bit flags for symbol_info().filling_mode
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2

REQUIRED_MT5_NAMES = (
    "initialize", "shutdown", "last_error", "symbol_select", "symbol_info", "symbol_info_tick",
    "copy_rates_from_pos", "positions_get", "account_info", "terminal_info", "order_send",
    "TRADE_ACTION_DEAL", "TRADE_ACTION_SLTP", "ORDER_TYPE_BUY", "ORDER_TYPE_SELL",
    "ORDER_TIME_GTC", "ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "ORDER_FILLING_RETURN",
    "POSITION_TYPE_BUY", "TRADE_RETCODE_DONE",
    "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_H1", "TIMEFRAME_H4",
    "TIMEFRAME_D1", "TIMEFRAME_W1",
)

MAX_OPEN_POSITIONS = 2   # 1 primary + 1 pyramid, or 2 primaries
MAX_TICK_AGE_SEC = 15    # P1: Strict tick freshness for gold (15 seconds)

# Master lock for all MT5 C-API calls
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
            self._last = now - self._refresh_sec + 60
            return
        diff = tick.time - now
        off = round(diff / 3600)
        if abs(diff - off * 3600) > 180 or not (-12 <= off <= 14):
            self._last = now - self._refresh_sec + 60
            return
        if off != self.offset or not self.verified:
            logger.info(f"Broker UTC offset set to {off:+d}h (verified from tick).")
        self.offset = off
        self.verified = True

    def tick_age(self, tick):
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
    source: str = "?"


class TradingState:
    def __init__(self):
        self.macro_bias = "NEUTRAL"
        self.macro_zones = []
        self.macro_in_poi_h4 = False
        self.macro_ts = 0.0

        self._sweep = None
        self.session = {"asian_high": 0.0, "asian_low": 0.0, "vol_ratio": 0.0,
                        "in_killzone": False, "vwap": 0.0, "broker_utc_offset": 0}

        self.agent_statuses = {}
        self.agent_beats = {}
        self.snapshot = {"market": {}, "account": {}, "trades": []}
        self.snapshot_ts = 0.0
        self.funnel = {}
        self.levels = {}
        self.level_setup = None
        
        # P2: Inter-agent Priority Locking ('level' > 'swing' > 'pyramid')
        self.armed_priority = None

        self.state_lock = threading.Lock()

    def update_agent_status(self, agent_name, status):
        with self.state_lock:
            self.agent_statuses[agent_name] = status

    def beat(self, agent_name, interval):
        with self.state_lock:
            self.agent_beats[agent_name] = (time.time(), interval)

    def set_priority(self, priority):
        with self.state_lock:
            self.armed_priority = priority

    def get_priority(self):
        with self.state_lock:
            return self.armed_priority

    def update_macro(self, bias, zones, in_poi_h4):
        with self.state_lock:
            self.macro_bias = bias
            self.macro_zones = [dict(z) for z in zones]
            self.macro_in_poi_h4 = bool(in_poi_h4)
            self.macro_ts = time.time()

    def get_macro(self, max_age=600):
        with self.state_lock:
            fresh = (time.time() - self.macro_ts) <= max_age
            return self.macro_bias, bool(self.macro_zones), fresh

    def zone_touched(self, low, high, tol=0.0):
        with self.state_lock:
            for z in self.macro_zones:
                if z["low"] - tol <= high and z["high"] + tol >= low:
                    return dict(z)
        return None

    def update_session_telemetry(self, asian_high, asian_low, vol_ratio, in_kz, vwap, offset):
        with self.state_lock:
            self.session = {"asian_high": float(asian_high), "asian_low": float(asian_low),
                            "vol_ratio": float(vol_ratio), "in_killzone": bool(in_kz),
                            "vwap": float(vwap), "broker_utc_offset": int(offset)}

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

    def update_levels(self, levels):
        with self.state_lock:
            self.levels = {k: float(v) for k, v in levels.items()}

    def update_level_setup(self, setup):
        with self.state_lock:
            self.level_setup = None if setup is None else dict(setup)

    def count(self, key, n=1):
        with self.state_lock:
            self.funnel[key] = self.funnel.get(key, 0) + n

    def update_snapshot(self, market, account, trades):
        with self.state_lock:
            self.snapshot = {"market": market, "account": account, "trades": trades}
            self.snapshot_ts = time.time()

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
                "priority": self.armed_priority,
                "market": self.snapshot["market"],
                "account": self.snapshot["account"],
                "trades": self.snapshot["trades"],
                "snapshot_age": round(now - self.snapshot_ts) if self.snapshot_ts else None,
                "funnel": dict(self.funnel),
                "levels": dict(self.levels),
                "level_setup": None if self.level_setup is None else {
                    **{k: v for k, v in self.level_setup.items() if k != "expires_at"},
                    "expires_in_sec": max(0, round(self.level_setup["expires_at"] - now)),
                },
            }


shared_state = TradingState()


# ==========================================
# 1.5 SHARED PATTERN HELPERS
# ==========================================
def volume_ratio(df, j, n=20):
    if j < n:
        return 0.0
    ma = df['tick_volume'].iloc[j - n:j].mean()
    return float(df['tick_volume'].iloc[j] / ma) if ma > 0 else 0.0


def detect_sweep(df, j, bullish, swing_n=30, gap=3, vol_mult=1.5):
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

    def fetch_data(self, bars=100, closed_only=True, timeframe=None):
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe or self.timeframe, 0, bars)
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
# 3. MACRO AGENT (H4) - BIAS & FVG ZONES
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
        idx = idx[idx >= 2]  # Require at least 2 bars prior for FVG derivation

        n = len(df)
        zones = []
        for i in idx[::-1]:
            i = int(i)
            if (n - 1) - i > self.MAX_ZONE_AGE:
                break
            
            # P1 Fix: Define zone strictly as the Fair Value Gap (FVG)
            if bullish:
                z_high = float(df['low'].iloc[i])        # gap top
                z_low  = float(df['high'].iloc[i - 2])   # gap bottom
            else:
                z_high = float(df['low'].iloc[i - 2])    # gap top
                z_low  = float(df['high'].iloc[i])        # gap bottom

            if z_high <= z_low:
                continue

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
            self.name, f"Bias: {bias} | {len(zones)} FVG zone(s) | H4 in POI: {in_poi_h4}")


# ==========================================
# 4. KILLZONE AGENT (M15) - ASIAN RANGE SWEEPS
# ==========================================
class KillzoneAgent(BaseAgent):
    ASIAN_END_HOUR = 6
    MIN_ASIAN_BARS = 20
    KILLZONES = ((7, 10), (12, 15))

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
        df = self.fetch_data(bars=130)
        if df is None or len(df) < 60:
            return

        n = len(df)
        bar = df.iloc[-1]
        t_utc = df['time'] - pd.Timedelta(hours=offset)
        hour = t_utc.dt.hour
        date = t_utc.dt.date

        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['tick_volume']
        cum_v = vol.groupby(date).cumsum()
        cum_tpv = (tp * vol).groupby(date).cumsum()
        vwap = float((cum_tpv / np.maximum(cum_v, 1)).iloc[-1])

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
            shared_state.update_agent_status(self.name, "Sweep ignored (level already taken today)")
            return

        zone = shared_state.zone_touched(float(bar['low']), float(bar['high']))
        if zone is None:
            shared_state.update_agent_status(self.name, "Sweep ignored (outside H4 FVG)")
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
                    f"H4 FVG {zone['low']:.2f}-{zone['high']:.2f})")
        shared_state.update_agent_status(self.name, f"LATCHED: {direction} ({level:.2f})")


# ==========================================
# 5. EXECUTION BASE CLASS
# ==========================================
class ExecutionAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, magic,
                 use_dynamic_lot, fixed_lot, risk_pct,
                 rr=2.0, deviation=20, cooldown_sec=300,
                 min_sl_spread_mult=3.0, max_spread_points=50.0):
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
        self._day_key = None
        self._day_start_equity = None

    def _daily_loss_breaker(self):
        """P0: Disable trading if today's equity drawdown exceeds 3%."""
        acct = mt5.account_info()
        if acct is None:
            return True
        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self._day_key != today:
            self._day_key = today
            self._day_start_equity = acct.equity
        if self._day_start_equity is None or self._day_start_equity <= 0:
            self._day_start_equity = acct.equity
            return False
        dd = (self._day_start_equity - acct.equity) / self._day_start_equity
        if dd >= 0.03:
            logger.error(f"[{self.name}] Daily loss breaker tripped! Drawdown {dd*100:.2f}% >= 3%.")
            return True
        return False

    def has_open_position(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return True
        mine = [p for p in positions if p.magic in SYSTEM_MAGICS]
        if any(p.magic == self.magic for p in mine):
            return True
        return len(mine) >= MAX_OPEN_POSITIONS

    @staticmethod
    def _filling_mode(info):
        if info.filling_mode & SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if info.filling_mode & SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _skip(self, reason):
        logger.warning(f"[{self.name}] Trade skipped: {reason}")
        return False

    def calculate_lot_size(self, info, entry, sl, risk_pct=None):
        rp = self.risk_pct if risk_pct is None else risk_pct
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

        raw_lot = (acct.balance * (rp / 100.0)) / risk_per_lot
        lot = math.floor(raw_lot / step + 1e-9) * step

        if lot < vmin:
            logger.warning(f"[{self.name}] Risk-based lot {raw_lot:.4f} < min {vmin} "
                           f"(SL distance {abs(entry - sl):.2f}); skipping.")
            return None

        lot = min(lot, vmax)
        return round(round(lot / step) * step, 8)

    def execute_trade(self, direction, sl_raw, tag=None, risk_pct=None):
        is_buy = direction == BULL_SWEEP

        with mt5_lock:
            if self._daily_loss_breaker():
                return self._skip("Daily loss circuit breaker active")

            if self.has_open_position():
                return False

            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
            if info is None or tick is None:
                return self._skip(f"no symbol/tick info ({mt5.last_error()})")

            terminal = mt5.terminal_info()
            if terminal is None or not terminal.connected:
                return self._skip("MT5 terminal disconnected from broker")

            age = broker_clock.tick_age(tick)
            if age is not None and abs(age) > MAX_TICK_AGE_SEC:
                return self._skip(f"stale tick ({age:.0f}s off server time)")

            spread = tick.ask - tick.bid
            spread_pts = spread / info.point
            if self.max_spread_points is not None and spread_pts > self.max_spread_points:
                return self._skip(f"spread too wide ({spread_pts:.0f} pts > {self.max_spread_points})")

            entry = tick.ask if is_buy else tick.bid
            sl = sl_raw - spread if is_buy else sl_raw + spread

            if (is_buy and sl >= entry) or (not is_buy and sl <= entry):
                return self._skip("SL on wrong side of entry")

            risk = abs(entry - sl)
            min_dist = max(info.trade_stops_level * info.point,
                           spread * self.min_sl_spread_mult)
            if risk < min_dist:
                return self._skip(f"SL distance {risk:.2f} < min allowed {min_dist:.2f}")

            tp = entry + risk * self.rr if is_buy else entry - risk * self.rr
            lot = self.calculate_lot_size(info, entry, sl, risk_pct)
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


# ==========================================
# 5.1 SWING AGENT (H1)
# ==========================================
class SwingAgent(ExecutionAgent):
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

        # P2: Yield to LevelSweep if armed
        if shared_state.get_priority() == "level":
            shared_state.update_agent_status(self.name, "Yielding (LevelSweep has priority)")
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
            return

        shared_state.count("swing_h1_fvg_with_sweep")
        j, level, extreme, zone = found
        logger.info(f"[{self.name}] H1 sweep at {df['time'].iloc[j]} "
                    f"at H4 zone {zone['low']:.2f}-{zone['high']:.2f} + FVG confirmed.")
        if self.execute_trade(direction, sl_raw, tag=f"{self.name}:H1SWEEP"):
            shared_state.count("swing_h1_trades")
            self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================
# 5.2 CONTINUATION AGENT (M5 Pyramid - Replaces Micro M1)
# ==========================================
class ContinuationAgent(ExecutionAgent):
    """P2 Replacement for MicroAgent:
    Pyramids into an active, winning LevelSweep trade at an M5 FVG pullback once
    the original position is secured at Break-Even.
    """
    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(
                self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        # 1. Verify parent LevelSweep position is active and at Break-Even
        parent = None
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                for p in positions:
                    if p.magic == MAGIC_LEVEL:
                        is_buy = p.type == mt5.POSITION_TYPE_BUY
                        is_be = (p.sl >= p.price_open) if is_buy else (p.sl <= p.price_open)
                        if is_be:
                            parent = p
                            break

        if parent is None:
            shared_state.update_agent_status(self.name, "Idle (Waiting for 1R BE trade to pyramid)")
            return

        with mt5_lock:
            if self.has_open_position():
                shared_state.update_agent_status(self.name, "Idle (Pyramid already active)")
                return

        df = self.fetch_data(bars=20, timeframe=mt5.TIMEFRAME_M5)
        if df is None or len(df) < 5:
            return

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c3['time'] == self.last_bar_time:
            return
        self.last_bar_time = c3['time']

        is_buy = parent.type == mt5.POSITION_TYPE_BUY
        direction = BULL_SWEEP if is_buy else BEAR_SWEEP

        if is_buy:
            has_fvg = c1['high'] < c3['low'] and c2['close'] > c2['open'] and c3['close'] > c1['high']
            sl_raw = float(c1['low'])
        else:
            has_fvg = c1['low'] > c3['high'] and c2['close'] < c2['open'] and c3['close'] < c1['low']
            sl_raw = float(c1['high'])

        if has_fvg:
            logger.info(f"[{self.name}] M5 Continuation FVG confirmed for active BE position {parent.ticket}.")
            shared_state.count("pyramid_m5_triggered")
            if self.execute_trade(direction, sl_raw, tag="Pyramid:M5", risk_pct=0.5):
                shared_state.count("pyramid_trades")
                self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================
# 5.5 LEVEL SWEEP AGENT (PDH/PDL + London H/L)
# ==========================================
class LevelSweepAgent(ExecutionAgent):
    LONDON_HOURS = (7, 12)
    KILLZONES = ((7, 10), (12, 15))
    LOW_LEVELS = ("PDL", "LDNL")
    HIGH_LEVELS = ("PDH", "LDNH")
    HTF_REFRESH_SEC = 600

    def __init__(self, *args, allow_shorts=True, allow_counter_bias=False,
                 min_score=5, a_score=7, risk_a_pct=1.0, risk_b_pct=0.5,
                 pending_ttl_sec=2700, choch_lookback=6, max_risk_atr=3.0,
                 max_trades_per_day=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.allow_counter_bias = allow_counter_bias
        self.min_score = min_score
        self.a_score = a_score
        self.risk_a_pct = risk_a_pct
        self.risk_b_pct = risk_b_pct
        self.pending_ttl_sec = pending_ttl_sec
        self.choch_lookback = choch_lookback
        self.max_risk_atr = max_risk_atr
        self.max_trades_per_day = max_trades_per_day

        self._htf = {}
        self._htf_ts = 0.0
        self.pending = None
        self.last_m15_time = None
        self.last_m5_time = None
        self.trades_today = 0
        self.trades_date = None

    def _refresh_htf(self):
        now = time.time()
        if now - self._htf_ts < self.HTF_REFRESH_SEC:
            return
        d1 = self.fetch_data(bars=3, closed_only=False, timeframe=mt5.TIMEFRAME_D1)
        w1 = self.fetch_data(bars=2, closed_only=False, timeframe=mt5.TIMEFRAME_W1)
        htf = dict(self._htf)
        if d1 is not None and len(d1) >= 2:
            htf["PDH"] = float(d1['high'].iloc[-2])
            htf["PDL"] = float(d1['low'].iloc[-2])
            htf["DO"] = float(d1['open'].iloc[-1])
        if w1 is not None and len(w1) >= 1:
            htf["WO"] = float(w1['open'].iloc[-1])
        self._htf = htf
        self._htf_ts = now if htf else now - self.HTF_REFRESH_SEC + 30

    def _london_levels(self, df, offset):
        t = df['time'] - pd.Timedelta(hours=offset)
        hour, date = t.dt.hour, t.dt.date
        today = date.iloc[-1]
        sess = df[(date == today) & (hour >= self.LONDON_HOURS[0]) & (hour < self.LONDON_HOURS[1])]
        # P1 Fix: Require at least 30 M5 bars (2.5 hours)
        if int(hour.iloc[-1]) >= self.LONDON_HOURS[1] and len(sess) >= 30:
            return {"LDNH": float(sess['high'].max()), "LDNL": float(sess['low'].min())}
        return {}

    def _detect(self, df, levels, bias):
        n = len(df)
        bar = df.iloc[-1]
        offset = broker_clock.offset
        hour_now = int((bar['time'] - pd.Timedelta(hours=offset)).hour)

        prev_close = df['close'].shift(1)
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
        atr = float(tr.rolling(14).mean().iloc[-2])
        if not atr > 0:
            return None

        low_hits = [(k, levels[k]) for k in self.LOW_LEVELS
                    if k in levels and bar['low'] < levels[k] < bar['close']]
        high_hits = [(k, levels[k]) for k in self.HIGH_LEVELS
                     if k in levels and bar['close'] < levels[k] < bar['high']]

        if not low_hits and not high_hits:
            return None
        shared_state.count("lvl_sweeps_detected")
        if low_hits and high_hits:
            shared_state.count("lvl_ambiguous_bar")
            return None

        bullish = bool(low_hits)
        hits = low_hits if bullish else high_hits
        name, level = max(hits, key=lambda kv: (kv[1] - bar['low']) if bullish else (bar['high'] - kv[1]))
        direction = BULL_SWEEP if bullish else BEAR_SWEEP
        extreme = float(bar['low'] if bullish else bar['high'])
        aligned = (bias == "BULLISH") == bullish

        parts = ["sweep+1"]
        score = 1

        # Level stacking
        tol = 0.5 * atr
        cands = [v for k, v in levels.items() if k != name]
        cands.append(round(level / 25.0) * 25.0)  # Extended to $25 milestones
        stack = min(2, sum(1 for v in cands if abs(v - level) <= tol))
        if stack:
            score += stack
            parts.append(f"stack+{stack}")

        # Volume
        vr = volume_ratio(df, n - 1)
        vpts = 2 if vr >= 2.5 else (1 if vr >= 1.5 else 0)
        if vpts:
            score += vpts
            parts.append(f"vol{vr:.1f}x+{vpts}")

        # Rejection candle
        rng = float(bar['high'] - bar['low'])
        if rng > 0:
            close_pos = float((bar['close'] - bar['low']) / rng)
            if bullish and close_pos >= 0.6 and bar['close'] > bar['open']:
                score += 1; parts.append("reject+1")
            if (not bullish) and close_pos <= 0.4 and bar['close'] < bar['open']:
                score += 1; parts.append("reject+1")

        # H4 alignment & zone
        if aligned:
            score += 1; parts.append("bias+1")
            if shared_state.zone_touched(float(bar['low']), float(bar['high'])) is not None:
                score += 2; parts.append("H4zone+2")

        if any(a <= hour_now <= b for a, b in self.KILLZONES):
            score += 1; parts.append("kz+1")

        grade = "A" if score >= self.a_score else ("B" if score >= self.min_score else None)
        risk = self.risk_a_pct if grade == "A" else (self.risk_b_pct if grade == "B" else 0.0)

        # P2: Shadow-log counter-bias setups
        if not aligned:
            if grade:
                shared_state.count(f"lvl_shadow_counter_bias_{grade}")
                logger.info(f"[{self.name} SHADOW] Counter-bias {direction} of {name} {level:.2f} (score {score}, grade {grade})")
            if not self.allow_counter_bias:
                shared_state.count("lvl_rejected_counter_bias")
                return None

        if not bullish and not self.allow_shorts:
            return None

        breakdown = " ".join(parts)
        logger.info(f"[{self.name}] {direction} of {name} {level:.2f} (extreme {extreme:.2f}): "
                    f"score {score} [{breakdown}] -> {grade or 'no trade'}")
        if grade is None:
            shared_state.count("lvl_below_min_score")
            return None

        bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
        return {
            "direction": direction, "level_name": name, "level": float(level),
            "extreme": extreme, "score": score, "grade": grade, "risk": risk,
            "breakdown": breakdown, "atr": atr, "attempts": 0,
            "bar_time": bar['time'], "close_time": bar['time'] + bar_delta,
            "expires_at": time.time() + self.pending_ttl_sec,
        }

    def _choch(self, m5, p, bullish):
        """P1 Fix: Require that both the extreme and the breakout candle form strictly post-sweep."""
        post = m5[m5['time'] >= p['bar_time']]
        if len(post) < 3:
            return False
        e = int(post['low'].idxmin() if bullish else post['high'].idxmax())
        n = len(m5)
        if e >= n - 1 or e < 1:
            return False
        if m5['time'].iloc[e] < p['close_time']:
            return False

        lo = max(0, e - self.choch_lookback)
        if bullish:
            brk = m5['close'].iloc[-1] > m5['high'].iloc[lo:e].max()
        else:
            brk = m5['close'].iloc[-1] < m5['low'].iloc[lo:e].min()

        return bool(brk and m5['time'].iloc[-1] > p['close_time'])

    def _check_trigger(self):
        p = self.pending
        m5 = self.fetch_data(bars=60)
        if m5 is None or len(m5) < 20:
            return
        last_t = m5['time'].iloc[-1]
        if last_t == self.last_m5_time:
            return
        self.last_m5_time = last_t

        bullish = p['direction'] == BULL_SWEEP
        c1, c2, c3 = m5.iloc[-3], m5.iloc[-2], m5.iloc[-1]
        close = float(c3['close'])

        if (bullish and close < p['extreme']) or ((not bullish) and close > p['extreme']):
            logger.info(f"[{self.name}] Setup failed (M5 close beyond sweep extreme); disarmed.")
            shared_state.count("lvl_setup_failed")
            self._disarm()
            return

        if bullish:
            fvg = c1['high'] < c3['low'] and c2['close'] > c2['open'] and c3['close'] > c1['high']
        else:
            fvg = c1['low'] > c3['high'] and c2['close'] < c2['open'] and c3['close'] < c1['low']
        fvg = bool(fvg and c2['time'] >= p['close_time'])
        choch = self._choch(m5, p, bullish)

        if not (fvg or choch):
            return

        trigger = "FVG" if fvg else "CHoCH"
        shared_state.count("lvl_trigger_fvg" if fvg else "lvl_trigger_choch")

        if abs(close - p['extreme']) > self.max_risk_atr * p['atr']:
            logger.info(f"[{self.name}] {trigger} trigger skipped: SL distance > {self.max_risk_atr} ATR.")
            shared_state.count("lvl_risk_too_wide")
            self._disarm()
            return

        logger.info(f"[{self.name}] {trigger} trigger after {p['level_name']} sweep (score {p['score']}).")
        tag = f"LvlSweep:{p['level_name']}:{p['grade']}{p['score']}"
        if self.execute_trade(p['direction'], p['extreme'], tag=tag, risk_pct=p['risk']):
            shared_state.count(f"lvl_trades_grade_{p['grade']}")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                logger.info(f"[{self.name}] Execution failed twice; setup disarmed.")
                self._disarm()

    def _disarm(self):
        self.pending = None
        shared_state.update_level_setup(None)
        if shared_state.get_priority() == "level":
            shared_state.set_priority(None)

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(
                self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            shared_state.update_agent_status(
                self.name, f"Daily trade cap reached ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        bias, has_zone, fresh = shared_state.get_macro()
        if bias not in ("BULLISH", "BEARISH") or not fresh:
            shared_state.update_agent_status(self.name, "Idle (Macro data stale)")
            return

        with mt5_lock:
            busy = self.has_open_position()
        if busy:
            shared_state.update_agent_status(self.name, "Idle (Position open / cap reached)")
            return

        if self.pending is not None and time.time() > self.pending['expires_at']:
            logger.info(f"[{self.name}] Armed setup expired.")
            shared_state.count("lvl_setup_expired")
            self._disarm()

        self._refresh_htf()
        m15 = self.fetch_data(bars=130, timeframe=mt5.TIMEFRAME_M15)
        if m15 is None or len(m15) < 60:
            return

        levels = dict(self._htf)
        levels.update(self._london_levels(m15, broker_clock.offset))
        shared_state.update_levels(levels)

        bar_time = m15['time'].iloc[-1]
        if bar_time != self.last_m15_time:
            self.last_m15_time = bar_time
            setup = self._detect(m15, levels, bias)
            if setup is not None and (self.pending is None or setup['score'] > self.pending['score']):
                self.pending = setup
                shared_state.set_priority("level")
                shared_state.count("lvl_setups_armed")

        if self.pending is None:
            shared_state.update_level_setup(None)
            shared_state.update_agent_status(self.name, f"Scanning levels ({len(levels)} tracked)")
            return

        p = self.pending
        shared_state.update_level_setup({
            "direction": p['direction'], "level_name": p['level_name'], "level": p['level'],
            "score": p['score'], "grade": p['grade'], "breakdown": p['breakdown'],
            "expires_at": p['expires_at'],
        })
        shared_state.update_agent_status(
            self.name, f"ARMED {p['direction']} {p['level_name']} (score {p['score']}/{p['grade']})")

        self._check_trigger()


# ==========================================
# 6. TRADE MANAGER (1R PARTIALS & BREAK-EVEN)
# ==========================================
class TradeManagerAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval,
                 rr=2.0, be_trigger_r=1.0, be_buffer_points=20):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.rr = rr
        self.be_trigger_r = be_trigger_r
        self.be_buffer_points = be_buffer_points
        self._risks = {}
        self._partials_taken = set()

    def step(self):
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            info = mt5.symbol_info(self.symbol)
        if positions is None or info is None:
            return

        mine = [p for p in positions if p.magic in SYSTEM_MAGICS]
        live = {p.ticket for p in mine}
        shared_state.update_agent_status(
            self.name, f"Managing {len(mine)} open position(s)")

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

            # Check 1R trigger threshold
            if profit < (risk * self.be_trigger_r):
                continue

            # P1: Scale out 50% partial volume at 1R
            if pos.ticket not in self._partials_taken:
                half_vol = math.floor((pos.volume / 2.0) / info.volume_step + 1e-9) * info.volume_step
                half_vol = round(round(half_vol / info.volume_step) * info.volume_step, 8)
                if half_vol >= info.volume_min:
                    close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
                    close_price = info.bid if is_buy else info.ask
                    partial_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "position": pos.ticket,
                        "volume": half_vol,
                        "type": close_type,
                        "price": close_price,
                        "deviation": 20,
                        "magic": pos.magic,
                        "comment": "Partials@1R",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": ExecutionAgent._filling_mode(info),
                    }
                    with mt5_lock:
                        p_res = mt5.order_send(partial_req)
                    if p_res and p_res.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"[{self.name}] Ticket {pos.ticket} 50% partial closed ({half_vol} lots).")
                        self._partials_taken.add(pos.ticket)

            # Move SL to Break-Even (+ buffer)
            secured = (pos.sl >= pos.price_open) if is_buy else (pos.sl <= pos.price_open)
            if secured:
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

            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[{self.name}] Ticket {pos.ticket} secured at break-even ({new_sl}).")

        self._risks = {t: v for t, v in self._risks.items() if t in live}
        self._partials_taken = {t for t in self._partials_taken if t in live}


# ==========================================
# 7. TELEMETRY AGENT
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
# 8. WEB DASHBOARD (FASTAPI)
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
  <div class="card"><h2>H4 Macro (FVG)</h2><div id="macro"></div></div>
  <div class="card"><h2>Killzones &amp; Liquidity (M15)</h2><div id="session"></div></div>
  <div class="card"><h2>Level sweeps (PDH/PDL &middot; London)</h2><div id="levels"></div></div>
  <div class="card"><h2>Setup funnel (telemetry)</h2><div id="funnel"></div></div>
  <div class="card wide"><h2>Agents &amp; Priority</h2><div id="agents"></div></div>
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
      ? zones.map((z, i) => row('FVG ' + (i + 1), `${num(z.low)} - ${num(z.high)} (${z.age} bars)`)).join('')
      : row('Zones', 'none valid', 'warn'));

  const sw = d.sweep;
  $('session').innerHTML =
    row('Killzone', s.in_killzone ? '<span class="badge on">OPEN</span>' : '<span class="badge off">CLOSED</span>') +
    row('Asian range', s.asian_high ? `${num(s.asian_low)} - ${num(s.asian_high)}` : 'none today') +
    row('Daily VWAP', s.vwap ? num(s.vwap) : '--') +
    row('Volume ratio', num(s.vol_ratio) + 'x') +
    row('Broker UTC offset', (s.broker_utc_offset >= 0 ? '+' : '') + s.broker_utc_offset + 'h') +
    row('Latched sweep', sw ? esc(sw.direction) + ' [' + esc(sw.source) + ']' : 'none', sw ? 'warn' : '');

  const lv = d.levels || {}, ls = d.level_setup;
  $('levels').innerHTML =
    (Object.keys(lv).length ? Object.keys(lv).map(k => row(k, num(lv[k]))).join('')
                            : '<div class="empty">no levels yet</div>') +
    (ls ? row('Armed setup', esc(ls.direction) + ' @ ' + esc(ls.level_name), 'warn') +
          row('Score / grade', esc(ls.score) + ' / ' + esc(ls.grade)) +
          row('Confluence', esc(ls.breakdown)) +
          row('Expires in', ls.expires_in_sec + 's')
        : row('Armed setup', 'none'));

  const f = d.funnel || {};
  const fk = Object.keys(f).sort();
  $('funnel').innerHTML = fk.length ? fk.map(k => row(k.replace(/_/g, ' '), f[k])).join('')
                                    : '<div class="empty">no setups recorded yet</div>';

  const names = Object.keys(d.agents || {});
  $('agents').innerHTML =
    row('Priority Lock', d.priority ? `<span class="badge on">${esc(d.priority.toUpperCase())}</span>` : 'NONE') +
    '<table><tr><th></th><th>Agent</th><th>Status</th><th>Last loop</th></tr>' +
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
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    except Exception:
        logger.error("Web server error", exc_info=True)


# ==========================================
# 9. ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    missing = [n for n in REQUIRED_MT5_NAMES if not hasattr(mt5, n)]
    if missing:
        logger.error(f"Installed MetaTrader5 package missing: {missing}.")
        quit()

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
                       max_spread_points=50.0, allow_shorts=ALLOW_SHORTS,
                       sweep_lookback=6, swing_n=30)
    levelsweep = LevelSweepAgent(
        "LevelSweep", SYMBOL, mt5.TIMEFRAME_M5, sleep_interval=10,
        magic=MAGIC_LEVEL, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=1.0,
        rr=RR, deviation=20, cooldown_sec=900, min_sl_spread_mult=3.0,
        max_spread_points=50.0,
        allow_shorts=ALLOW_SHORTS, allow_counter_bias=False,
        min_score=5, a_score=7, risk_a_pct=1.0, risk_b_pct=0.5,
        pending_ttl_sec=2700, choch_lookback=6, max_risk_atr=3.0, max_trades_per_day=4,
    )
    continuation = ContinuationAgent(
        "Continuation_M5", SYMBOL, mt5.TIMEFRAME_M5, sleep_interval=5,
        magic=MAGIC_PYRAMID, use_dynamic_lot=True, fixed_lot=0.05, risk_pct=0.5,
        rr=2.0, deviation=20, cooldown_sec=600, min_sl_spread_mult=3.0,
        max_spread_points=40.0
    )
    manager = TradeManagerAgent("Manager", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=3,
                                rr=RR, be_trigger_r=1.0, be_buffer_points=20)
    telemetry = TelemetryAgent("Telemetry", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2)

    agents = [macro, killzone, swing, levelsweep, continuation, manager, telemetry]
    for a in agents:
        a.start()

    threading.Thread(target=run_web_server, name="WebServer", daemon=True).start()

    warned = set()
    terminal_was_ok = True
    try:
        logger.info("System online. Dashboard: http://127.0.0.1:8000 (Ctrl+C to stop)")
        while True:
            time.sleep(5)
            with mt5_lock:
                terminal = mt5.terminal_info()
            if terminal is None:
                logger.critical("!! FATAL: MT5 terminal closed or unreachable. !!")
                break
            if not terminal.connected and terminal_was_ok:
                logger.error("!! MT5 lost broker connection: trades blocked. !!")
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
                    logger.warning(f"!! [{a.name}] loop delay: {time.time() - a.last_beat:.0f}s.")
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