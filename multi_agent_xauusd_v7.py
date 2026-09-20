# ==========================================================================
# multi_agent_xauusd_v8.py
# ==========================================================================
import datetime
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

# ==========================================================================
# 0. CONSTANTS, LOCKS & LOGGER
# ==========================================================================
SYMBOL = "XAUUSD.sd"

MAGIC_SWING     = 999998
MAGIC_LEVEL     = 999997
MAGIC_PYRAMID   = 999996
MAGIC_FAILBO    = 999995
MAGIC_VWAP      = 999994
SYSTEM_MAGICS = {MAGIC_SWING, MAGIC_LEVEL, MAGIC_PYRAMID, MAGIC_FAILBO, MAGIC_VWAP}

BULL_SWEEP = "BULLISH_SWEEP"
BEAR_SWEEP = "BEARISH_SWEEP"

BROKER_UTC_OFFSET_FALLBACK = 3

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

MAX_OPEN_POSITIONS = 2
MAX_TICK_AGE_SEC = 15

# --- Trading hours (UTC) ---
NO_ENTRY_BEFORE_UTC = 1     # 01:00 UTC - skip dead Asian hours
NO_ENTRY_AFTER_UTC  = 19    # 19:00 UTC - no new entries
FLAT_AT_UTC         = 20    # 20:30 UTC - force close everything
FLAT_AT_MINUTE      = 30

# --- Daily risk ---
DAILY_LOSS_CAP_PCT = 1.75   # stop trading for the day at -1.75% equity

# --- Risk per grade ---
RISK_A_PCT      = 0.75
RISK_B_PCT      = 0.50
RISK_PYRAMID    = 0.40

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


# ==========================================================================
# 0.5 BROKER CLOCK
# ==========================================================================
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


# ==========================================================================
# 0.7 TIME-OF-DAY HELPERS
# ==========================================================================
def now_utc():
    return datetime.datetime.utcnow()

def utc_minutes():
    n = now_utc()
    return n.hour * 60 + n.minute

def in_trading_hours():
    h = now_utc().hour
    return NO_ENTRY_BEFORE_UTC <= h < NO_ENTRY_AFTER_UTC

def is_flat_time():
    n = now_utc()
    if n.hour > FLAT_AT_UTC:
        return True
    if n.hour == FLAT_AT_UTC and n.minute >= FLAT_AT_MINUTE:
        return True
    return False

def news_blackout():
    """Return True if we're in a known high-impact news blackout window.
    Conservative — blocks a wider window than strictly necessary; update
    manually around FOMC dates."""
    n = now_utc()
    m = n.hour * 60 + n.minute
    # NFP: first Friday of month, 13:30 UTC. Block 13:15-14:15.
    if n.weekday() == 4 and n.day <= 7 and (13*60+15) <= m <= (14*60+15):
        return True
    # CPI (roughly 2nd/3rd Tue-Wed, 13:30 UTC). Block 13:15-14:15.
    if 8 <= n.day <= 15 and n.weekday() in (1, 2) and (13*60+15) <= m <= (14*60+15):
        return True
    # Daily 15:00 UTC (10:00 ET) - brief blackout
    if (14*60+55) <= m <= (15*60+15):
        return True
    return False


# ==========================================================================
# 1. SHARED STATE & TELEMETRY
# ==========================================================================
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
                        "in_killzone": False, "vwap": 0.0, "broker_utc_offset": 0,
                        "h1_atr": 0.0}

        self.agent_statuses = {}
        self.agent_beats = {}
        self.snapshot = {"market": {}, "account": {}, "trades": []}
        self.snapshot_ts = 0.0
        self.funnel = {}
        self.levels = {}
        self.level_setup = None
        self.failbo_setup = None
        self.vwap_setup = None
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

    def update_session_telemetry(self, **kwargs):
        with self.state_lock:
            self.session.update(kwargs)

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

    def update_failbo_setup(self, setup):
        with self.state_lock:
            self.failbo_setup = None if setup is None else dict(setup)

    def update_vwap_setup(self, setup):
        with self.state_lock:
            self.vwap_setup = None if setup is None else dict(setup)

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

            def setup_dict(s):
                if s is None:
                    return None
                d = {k: v for k, v in s.items() if k != "expires_at"}
                if "expires_at" in s:
                    d["expires_in_sec"] = max(0, round(s["expires_at"] - now))
                return d

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
                "level_setup": setup_dict(self.level_setup),
                "failbo_setup": setup_dict(self.failbo_setup),
                "vwap_setup": setup_dict(self.vwap_setup),
                "news_blackout": news_blackout(),
                "trading_hours_open": in_trading_hours(),
            }


shared_state = TradingState()


# ==========================================================================
# 1.5 SHARED PATTERN HELPERS
# ==========================================================================
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


def compute_atr(df, period=14):
    if len(df) < period + 2:
        return 0.0
    prev_close = df['close'].shift(1)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - prev_close).abs(),
                               (df['low'] - prev_close).abs()))
    val = float(tr.rolling(period).mean().iloc[-1])
    return val if val > 0 else 0.0


# ==========================================================================
# 2. BASE AGENT
# ==========================================================================
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


# ==========================================================================
# 3. MACRO AGENT (H4)
# ==========================================================================
class MacroAgent(BaseAgent):
    EXPANSION_MULT = 1.5
    ATR_PERIOD = 14
    MAX_ZONE_AGE = 100
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
        idx = idx[idx >= 2]

        n = len(df)
        zones = []
        for i in idx[::-1]:
            i = int(i)
            if (n - 1) - i > self.MAX_ZONE_AGE:
                break
            if bullish:
                z_high = float(df['low'].iloc[i])
                z_low  = float(df['high'].iloc[i - 2])
            else:
                z_high = float(df['low'].iloc[i - 2])
                z_low  = float(df['high'].iloc[i])
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


# ==========================================================================
# 4. KILLZONE AGENT (M15)
# ==========================================================================
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
        h1 = self.fetch_data(bars=20, timeframe=mt5.TIMEFRAME_H1)
        h1_atr = compute_atr(h1, 14) if h1 is not None else 0.0

        shared_state.update_session_telemetry(
            asian_high=self.asian_high, asian_low=self.asian_low,
            vol_ratio=vol_ratio, in_killzone=in_kz, vwap=vwap,
            broker_utc_offset=offset, h1_atr=h1_atr)

        if bar['time'] == self.last_bar_time:
            return
        self.last_bar_time = bar['time']

        if bias not in ("BULLISH", "BEARISH") or not fresh:
            shared_state.update_agent_status(self.name, "Idle (Macro data stale)")
            return
        if not has_zone:
            shared_state.update_agent_status(self.name, "Idle (No valid H4 zone)")
            return
        if not in_kz:
            shared_state.update_agent_status(
                self.name, f"Outside KZ (Asian {self.asian_low:.1f}-{self.asian_high:.1f})")
            return
        if self.asian_high <= 0 or self.asian_low <= 0:
            shared_state.update_agent_status(self.name, "Idle (No Asian range today)")
            return
        if shared_state.get_sweep() is not None:
            shared_state.update_agent_status(self.name, "Sweep latched")
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
                self.name, f"Scanning KZ | Vol {vol_ratio:.2f}x")
            return

        shared_state.count("kz_sweep_pattern")
        if already_taken:
            shared_state.update_agent_status(self.name, "Level already taken today")
            return

        zone = shared_state.zone_touched(float(bar['low']), float(bar['high']))
        if zone is None:
            shared_state.update_agent_status(self.name, "Sweep outside H4 FVG")
            return

        shared_state.count("kz_fresh_and_at_h4_zone")
        bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
        shared_state.set_sweep(Sweep(
            direction=direction, level=level, extreme=extreme,
            bar_time=bar['time'], close_time=bar['time'] + bar_delta,
            expires_at=time.time() + self.sweep_ttl_sec, source="ASIAN",
        ))
        logger.info(f"[{self.name}] {direction} latched at {bar['time']} "
                    f"(level {level:.2f}, extreme {extreme:.2f}, "
                    f"H4 FVG {zone['low']:.2f}-{zone['high']:.2f})")
        shared_state.update_agent_status(self.name, f"LATCHED: {direction}")


# ==========================================================================
# 5. EXECUTION BASE CLASS
# ==========================================================================
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
        self._atr_cache = 0.0
        self._atr_cache_ts = 0.0

    # ---------- risk ----------
    def _daily_loss_breaker(self):
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
        if dd * 100 >= DAILY_LOSS_CAP_PCT:
            return True
        return False

    def _h1_atr(self):
        now = time.time()
        if now - self._atr_cache_ts < 300 and self._atr_cache > 0:
            return self._atr_cache
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 20)
        if rates is None or len(rates) < 15:
            return self._atr_cache or 0.0
        df = pd.DataFrame(rates).iloc[:-1]
        self._atr_cache = compute_atr(df, 14)
        self._atr_cache_ts = now
        return self._atr_cache

    # ---------- position checks ----------
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
            return None

        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value_loss or info.trade_tick_value
        if tick_size <= 0 or tick_value <= 0:
            return None

        risk_per_lot = (abs(entry - sl) / tick_size) * tick_value
        if risk_per_lot <= 0:
            return None

        raw_lot = (acct.balance * (rp / 100.0)) / risk_per_lot
        lot = math.floor(raw_lot / step + 1e-9) * step

        if lot < vmin:
            logger.warning(f"[{self.name}] Lot {raw_lot:.4f} < min {vmin}; skipped.")
            return None

        lot = min(lot, vmax)
        return round(round(lot / step) * step, 8)

    # ---------- order execution ----------
    def execute_trade(self, direction, sl_raw, tag=None, risk_pct=None,
                      rr_override=None, max_risk_override=None):
        is_buy = direction == BULL_SWEEP

        # Time-of-day gates (outside lock — no MT5 calls)
        if not in_trading_hours():
            return self._skip("outside trading hours (no-entry window)")
        if news_blackout():
            return self._skip("news blackout window")

        with mt5_lock:
            if self._daily_loss_breaker():
                return self._skip("daily loss breaker active")

            if self.has_open_position():
                return False

            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
            if info is None or tick is None:
                return self._skip(f"no symbol/tick ({mt5.last_error()})")

            terminal = mt5.terminal_info()
            if terminal is None or not terminal.connected:
                return self._skip("terminal disconnected")

            age = broker_clock.tick_age(tick)
            if age is not None and abs(age) > MAX_TICK_AGE_SEC:
                return self._skip(f"stale tick ({age:.0f}s)")

            spread = tick.ask - tick.bid
            spread_pts = spread / info.point
            if self.max_spread_points is not None and spread_pts > self.max_spread_points:
                return self._skip(f"spread {spread_pts:.0f} pts > {self.max_spread_points}")

            entry = tick.ask if is_buy else tick.bid
            sl = sl_raw - spread if is_buy else sl_raw + spread

            if (is_buy and sl >= entry) or (not is_buy and sl <= entry):
                return self._skip("SL wrong side")

            risk = abs(entry - sl)
            min_dist = max(info.trade_stops_level * info.point,
                           spread * self.min_sl_spread_mult)
            if risk < min_dist:
                return self._skip(f"SL {risk:.2f} < min {min_dist:.2f}")

            if max_risk_override is not None and risk > max_risk_override:
                return self._skip(f"risk {risk:.2f} > cap {max_risk_override:.2f}")

            # ATR-aware TP; set far away so trailing can run
            rr_used = rr_override if rr_override is not None else self.rr
            atr = self._h1_atr()
            target_dist = risk * rr_used
            if atr > 0:
                target_dist = max(target_dist, 0.6 * atr)
            # Cap so we don't set absurd TPs
            target_dist = min(target_dist, risk * 6.0)
            tp = entry + target_dist if is_buy else entry - target_dist

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
            logger.warning(f"[{self.name}] order_send None: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"[{self.name}] Rejected: {result.retcode} {result.comment}")
            return False

        logger.info(f"[{self.name}] Filled {'BUY' if is_buy else 'SELL'} {lot} @ "
                    f"{result.price}. SL {request['sl']}, TP {request['tp']} [{request['comment']}]")
        return True


# ==========================================================================
# 5.1 SWING AGENT (H1)
# ==========================================================================
class SwingAgent(ExecutionAgent):
    def __init__(self, *args, allow_shorts=False, sweep_lookback=6, swing_n=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.sweep_lookback = sweep_lookback
        self.swing_n = swing_n

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return
        prio = shared_state.get_priority()
        if prio in ("level", "failbo"):
            shared_state.update_agent_status(self.name, f"Yielding ({prio} has priority)")
            return

        bias, has_zone, fresh = shared_state.get_macro()
        if not (fresh and has_zone) or bias not in ("BULLISH", "BEARISH"):
            shared_state.update_agent_status(self.name, "Idle (No valid H4 zone)")
            return
        if bias == "BEARISH" and not self.allow_shorts:
            shared_state.update_agent_status(self.name, "Idle (Shorts disabled)")
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
        shared_state.update_agent_status(self.name, f"Hunting H1 sweep+FVG ({bias})")

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
        logger.info(f"[{self.name}] H1 sweep {df['time'].iloc[j]} @ H4 {zone['low']:.2f}-{zone['high']:.2f} + FVG.")
        if self.execute_trade(direction, sl_raw, tag=f"Swing:H1", rr_override=2.0):
            shared_state.count("swing_h1_trades")
            self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================================================
# 5.2 CONTINUATION AGENT (M15 pyramid, gated on parent 1.2R)
# ==========================================================================
class ContinuationAgent(ExecutionAgent):
    PARENT_MIN_PROFIT_R = 1.2

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        # Find a parent position that is BE-secured AND >= 1.2R in profit
        parent = None
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                shared_state.update_agent_status(self.name, "Idle (positions unavailable)")
                return
            for p in positions:
                if p.magic != MAGIC_LEVEL:
                    continue
                is_buy = p.type == mt5.POSITION_TYPE_BUY
                is_be = (p.sl >= p.price_open) if is_buy else (p.sl <= p.price_open)
                if not is_be:
                    continue
                risk = abs(p.price_open - p.sl)
                if risk <= 0:
                    continue
                profit = (p.price_current - p.price_open) if is_buy \
                         else (p.price_open - p.price_current)
                if profit >= risk * self.PARENT_MIN_PROFIT_R:
                    parent = p
                    parent_risk = risk
                    break

        if parent is None:
            shared_state.update_agent_status(self.name, f"Idle (no parent >= {self.PARENT_MIN_PROFIT_R}R)")
            return

        with mt5_lock:
            if self.has_open_position():
                shared_state.update_agent_status(self.name, "Idle (Pyramid already open)")
                return

        # M15 FVG trigger
        m15 = self.fetch_data(bars=20, timeframe=mt5.TIMEFRAME_M15)
        if m15 is None or len(m15) < 5:
            return
        c1, c2, c3 = m15.iloc[-3], m15.iloc[-2], m15.iloc[-1]
        if c3['time'] == self.last_bar_time:
            return
        self.last_bar_time = c3['time']

        is_buy = parent.type == mt5.POSITION_TYPE_BUY
        direction = BULL_SWEEP if is_buy else BEAR_SWEEP

        if is_buy:
            has_fvg = c1['high'] < c3['low'] and c2['close'] > c2['open'] and c3['close'] > c1['high']
            structural_sl = float(c1['low'])
            # Never below parent's SL (i.e., never wider risk on the parent thesis)
            sl_raw = max(structural_sl, parent.sl)
        else:
            has_fvg = c1['low'] > c3['high'] and c2['close'] < c2['open'] and c3['close'] < c1['low']
            structural_sl = float(c1['high'])
            sl_raw = min(structural_sl, parent.sl)

        if not has_fvg:
            shared_state.update_agent_status(self.name, f"Waiting M15 FVG for {direction}")
            return

        # Final sanity: pyramid risk must not exceed 1.5x parent risk
        with mt5_lock:
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return
        entry = tick.ask if is_buy else tick.bid
        pyramid_risk = abs(entry - sl_raw)
        max_allowed = parent_risk * 1.5
        if pyramid_risk > max_allowed:
            logger.info(f"[{self.name}] Pyramid SL {pyramid_risk:.2f} > "
                        f"1.5x parent risk {max_allowed:.2f}; skipped.")
            shared_state.count("pyramid_risk_rejected")
            return

        logger.info(f"[{self.name}] M15 FVG for pyramid (parent ticket {parent.ticket}).")
        shared_state.count("pyramid_m15_triggered")
        if self.execute_trade(direction, sl_raw, tag="Pyr:M15",
                              risk_pct=RISK_PYRAMID, rr_override=2.5,
                              max_risk_override=max_allowed):
            shared_state.count("pyramid_trades")
            self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================================================
# 5.5 LEVEL SWEEP AGENT (PDH/PDL + London H/L + $25/$50 round numbers)
# ==========================================================================
class LevelSweepAgent(ExecutionAgent):
    LONDON_HOURS = (7, 12)
    KILLZONES = ((7, 10), (12, 15))
    LOW_LEVELS = ("PDL", "LDNL")
    HIGH_LEVELS = ("PDH", "LDNH")
    HTF_REFRESH_SEC = 600

    def __init__(self, *args, allow_shorts=True, allow_counter_bias=False,
                 min_score=5, a_score=7,
                 pending_ttl_sec=2700, choch_lookback=6, max_risk_atr=3.0,
                 max_trades_per_day=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.allow_counter_bias = allow_counter_bias
        self.min_score = min_score
        self.a_score = a_score
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
        if int(hour.iloc[-1]) >= self.LONDON_HOURS[1] and len(sess) >= 30:
            return {"LDNH": float(sess['high'].max()), "LDNL": float(sess['low'].min())}
        return {}

    def _round_number_levels(self, price):
        base = round(price / 25.0) * 25.0
        return {f"R{int(base + i*25)}": base + i * 25.0 for i in range(-2, 3)}

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

        low_hits = [(k, levels[k]) for k in levels
                    if k in self.LOW_LEVELS + tuple(k for k in levels if k.startswith("R"))
                    and bar['low'] < levels[k] < bar['close']]
        # Filter for HIGH_LEVELS and round numbers above
        high_hits = [(k, levels[k]) for k in levels
                     if k in self.HIGH_LEVELS + tuple(k for k in levels if k.startswith("R"))
                     and bar['close'] < levels[k] < bar['high']]

        if not low_hits and not high_hits:
            return None
        shared_state.count("lvl_sweeps_detected")
        if low_hits and high_hits:
            shared_state.count("lvl_ambiguous_bar")
            return None

        bullish = bool(low_hits)
        hits = low_hits if bullish else high_hits
        # Prefer named levels (PDH/PDL etc) over round numbers for tie-break
        def priority(kv):
            k, v = kv
            named = 1 if (k in self.LOW_LEVELS or k in self.HIGH_LEVELS) else 0
            depth = (v - bar['low']) if bullish else (bar['high'] - v)
            return (named, depth)
        name, level = max(hits, key=priority)
        direction = BULL_SWEEP if bullish else BEAR_SWEEP
        extreme = float(bar['low'] if bullish else bar['high'])
        aligned = (bias == "BULLISH") == bullish

        parts = ["sweep+1"]
        score = 1

        tol = 0.5 * atr
        cands = [v for k, v in levels.items() if k != name]
        stack = min(2, sum(1 for v in cands if abs(v - level) <= tol))
        if stack:
            score += stack
            parts.append(f"stack+{stack}")

        vr = volume_ratio(df, n - 1)
        vpts = 2 if vr >= 2.5 else (1 if vr >= 1.5 else 0)
        if vpts:
            score += vpts
            parts.append(f"vol{vr:.1f}x+{vpts}")

        rng = float(bar['high'] - bar['low'])
        if rng > 0:
            close_pos = float((bar['close'] - bar['low']) / rng)
            if bullish and close_pos >= 0.6 and bar['close'] > bar['open']:
                score += 1; parts.append("reject+1")
            if (not bullish) and close_pos <= 0.4 and bar['close'] < bar['open']:
                score += 1; parts.append("reject+1")

        if aligned:
            score += 1; parts.append("bias+1")
            if shared_state.zone_touched(float(bar['low']), float(bar['high'])) is not None:
                score += 2; parts.append("H4zone+2")

        if any(a <= hour_now <= b for a, b in self.KILLZONES):
            score += 1; parts.append("kz+1")

        grade = "A" if score >= self.a_score else ("B" if score >= self.min_score else None)
        risk = RISK_A_PCT if grade == "A" else (RISK_B_PCT if grade == "B" else 0.0)
        rr_used = 3.0 if grade == "A" else 2.0

        if not aligned:
            if grade:
                shared_state.count(f"lvl_shadow_counter_bias_{grade}")
                logger.info(f"[{self.name} SHADOW] Counter-bias {direction} {name} {level:.2f} "
                            f"(score {score}, grade {grade})")
            if not self.allow_counter_bias:
                shared_state.count("lvl_rejected_counter_bias")
                return None

        if not bullish and not self.allow_shorts:
            return None

        breakdown = " ".join(parts)
        logger.info(f"[{self.name}] {direction} {name} {level:.2f}: score {score} [{breakdown}] -> {grade or 'no trade'}")
        if grade is None:
            shared_state.count("lvl_below_min_score")
            return None

        bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
        return {
            "direction": direction, "level_name": name, "level": float(level),
            "extreme": extreme, "score": score, "grade": grade, "risk": risk,
            "rr": rr_used, "breakdown": breakdown, "atr": atr, "attempts": 0,
            "bar_time": bar['time'], "close_time": bar['time'] + bar_delta,
            "expires_at": time.time() + self.pending_ttl_sec,
        }

    def _choch(self, m5, p, bullish):
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
            logger.info(f"[{self.name}] Setup failed (M5 close beyond extreme); disarmed.")
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
            logger.info(f"[{self.name}] {trigger} SL too wide > {self.max_risk_atr} ATR.")
            shared_state.count("lvl_risk_too_wide")
            self._disarm()
            return

        logger.info(f"[{self.name}] {trigger} after {p['level_name']} sweep (score {p['score']}).")
        tag = f"Lvl:{p['level_name']}:{p['grade']}{p['score']}"
        if self.execute_trade(p['direction'], p['extreme'], tag=tag,
                              risk_pct=p['risk'], rr_override=p['rr']):
            shared_state.count(f"lvl_trades_grade_{p['grade']}")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                self._disarm()

    def _disarm(self):
        self.pending = None
        shared_state.update_level_setup(None)
        if shared_state.get_priority() == "level":
            shared_state.set_priority(None)

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            shared_state.update_agent_status(self.name, f"Daily cap reached ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        bias, has_zone, fresh = shared_state.get_macro()
        if bias not in ("BULLISH", "BEARISH") or not fresh:
            shared_state.update_agent_status(self.name, "Idle (Macro stale)")
            return

        with mt5_lock:
            busy = self.has_open_position()
        if busy:
            shared_state.update_agent_status(self.name, "Idle (Position/cap)")
            return

        if self.pending is not None and time.time() > self.pending['expires_at']:
            shared_state.count("lvl_setup_expired")
            self._disarm()

        self._refresh_htf()
        m15 = self.fetch_data(bars=130, timeframe=mt5.TIMEFRAME_M15)
        if m15 is None or len(m15) < 60:
            return

        levels = dict(self._htf)
        levels.update(self._london_levels(m15, broker_clock.offset))
        # Add $25 round numbers near current price
        price = float(m15['close'].iloc[-1])
        levels.update(self._round_number_levels(price))
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
            shared_state.update_agent_status(self.name, f"Scanning ({len(levels)} levels)")
            return

        p = self.pending
        shared_state.update_level_setup({
            "direction": p['direction'], "level_name": p['level_name'], "level": p['level'],
            "score": p['score'], "grade": p['grade'], "breakdown": p['breakdown'],
            "expires_at": p['expires_at'],
        })
        shared_state.update_agent_status(
            self.name, f"ARMED {p['direction']} {p['level_name']} ({p['score']}/{p['grade']})")
        self._check_trigger()


# ==========================================================================
# 5.6 FAILED BREAKOUT AGENT (counter-trend PDH/PDL fade)
# ==========================================================================
class FailedBreakoutAgent(ExecutionAgent):
    """PDH/PDL breaks during a killzone and fails to hold on H1 close.
    Enter on the M15 retest of the broken level in the reverse direction.
    This is a COUNTER-BIAS setup by design; it fires even when H4 bias disagrees."""
    KILLZONES = ((7, 10), (12, 15))
    HTF_REFRESH_SEC = 600
    MIN_BREAK_ATR = 0.3   # break must exceed level by >= 0.3 ATR(H1)

    def __init__(self, *args, allow_shorts=True, pending_ttl_sec=1800,
                 max_risk_atr=2.5, max_trades_per_day=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.pending_ttl_sec = pending_ttl_sec
        self.max_risk_atr = max_risk_atr
        self.max_trades_per_day = max_trades_per_day
        self._htf = {}
        self._htf_ts = 0.0
        self.pending = None
        self.last_h1_time = None
        self.last_m15_time = None
        self.trades_today = 0
        self.trades_date = None

    def _refresh_htf(self):
        now = time.time()
        if now - self._htf_ts < self.HTF_REFRESH_SEC:
            return
        d1 = self.fetch_data(bars=3, closed_only=False, timeframe=mt5.TIMEFRAME_D1)
        htf = dict(self._htf)
        if d1 is not None and len(d1) >= 2:
            htf["PDH"] = float(d1['high'].iloc[-2])
            htf["PDL"] = float(d1['low'].iloc[-2])
        self._htf = htf
        self._htf_ts = now if htf else now - self.HTF_REFRESH_SEC + 30

    def _scan_h1(self):
        h1 = self.fetch_data(bars=30, timeframe=mt5.TIMEFRAME_H1)
        if h1 is None or len(h1) < 20:
            return
        bar = h1.iloc[-1]
        if bar['time'] == self.last_h1_time:
            return
        self.last_h1_time = bar['time']

        atr = compute_atr(h1, 14)
        if atr <= 0:
            return
        offset = broker_clock.offset
        hour_now = int((bar['time'] - pd.Timedelta(hours=offset)).hour)
        if not any(a <= hour_now <= b for a, b in self.KILLZONES):
            return

        pdh = self._htf.get("PDH")
        pdl = self._htf.get("PDL")

        # Look at the last 3 H1 bars for a break-and-fail pattern
        for i in range(len(h1) - 3, len(h1)):
            b = h1.iloc[i]
            if pdh is not None:
                broke_above = b['high'] > pdh + self.MIN_BREAK_ATR * atr
                closed_below = b['close'] < pdh
                if broke_above and closed_below:
                    self._arm(BEAR_SWEEP, "PDH", pdh, float(b['high']), atr)
                    return
            if pdl is not None:
                broke_below = b['low'] < pdl - self.MIN_BREAK_ATR * atr
                closed_above = b['close'] > pdl
                if broke_below and closed_above:
                    self._arm(BULL_SWEEP, "PDL", pdl, float(b['low']), atr)
                    return

    def _arm(self, direction, level_name, level, extreme, atr):
        shared_state.count("failbo_setups_armed")
        close_time = now_utc()
        self.pending = {
            "direction": direction, "level_name": level_name,
            "level": float(level), "extreme": float(extreme),
            "atr": atr, "expires_at": time.time() + self.pending_ttl_sec,
            "attempts": 0,
        }
        shared_state.set_priority("failbo")
        logger.info(f"[{self.name}] ARMED {direction} fade of broken {level_name} {level:.2f} "
                    f"(extreme {extreme:.2f})")

    def _disarm(self):
        self.pending = None
        shared_state.update_failbo_setup(None)
        if shared_state.get_priority() == "failbo":
            shared_state.set_priority(None)

    def _check_trigger(self):
        p = self.pending
        m15 = self.fetch_data(bars=40, timeframe=mt5.TIMEFRAME_M15)
        if m15 is None or len(m15) < 10:
            return
        last_t = m15['time'].iloc[-1]
        if last_t == self.last_m15_time:
            return
        self.last_m15_time = last_t

        bullish = p['direction'] == BULL_SWEEP
        bar = m15.iloc[-1]
        close = float(bar['close'])

        # Invalidation: price reclaims the extreme
        if (bullish and close < p['extreme']) or ((not bullish) and close > p['extreme']):
            shared_state.count("failbo_failed")
            logger.info(f"[{self.name}] Failed breakout invalidated.")
            self._disarm()
            return

        # Trigger: M15 retest of the broken level (within 0.5 ATR H1)
        tol = 0.5 * p['atr']
        if bullish:
            touched = bar['low'] <= p['level'] + tol and close > p['level']
        else:
            touched = bar['high'] >= p['level'] - tol and close < p['level']
        if not touched:
            return

        # SL: sweep extreme. TP: opposite PDH/PDL or 2.5R
        sl_raw = p['extreme']
        with mt5_lock:
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return
        entry = tick.ask if bullish else tick.bid
        risk = abs(entry - sl_raw)
        if risk > self.max_risk_atr * p['atr']:
            shared_state.count("failbo_risk_too_wide")
            self._disarm()
            return

        logger.info(f"[{self.name}] M15 retest trigger for {p['level_name']} fade.")
        shared_state.count("failbo_trigger")
        tag = f"FailBO:{p['level_name']}"
        if self.execute_trade(p['direction'], sl_raw, tag=tag,
                              risk_pct=RISK_B_PCT, rr_override=2.5):
            shared_state.count("failbo_trades")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                self._disarm()

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            shared_state.update_agent_status(self.name, f"Daily cap ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        if self.pending is not None and time.time() > self.pending['expires_at']:
            shared_state.count("failbo_expired")
            self._disarm()

        self._refresh_htf()

        with mt5_lock:
            busy = self.has_open_position()
        if busy and self.pending is None:
            shared_state.update_agent_status(self.name, "Idle (Position/cap)")
            return

        self._scan_h1()

        if self.pending is None:
            shared_state.update_failbo_setup(None)
            shared_state.update_agent_status(self.name, "Scanning for failed breaks")
            return

        p = self.pending
        shared_state.update_failbo_setup({
            "direction": p['direction'], "level_name": p['level_name'],
            "level": p['level'], "extreme": p['extreme'],
            "expires_at": p['expires_at'],
        })
        shared_state.update_agent_status(
            self.name, f"ARMED fade {p['level_name']}")
        self._check_trigger()


# ==========================================================================
# 5.7 VWAP REVERSION AGENT (mean reversion during chop)
# ==========================================================================
class VWAPReversionAgent(ExecutionAgent):
    """Fade exhaustion moves back to daily VWAP.
    Only fires after 14:00 UTC, requires >=2.5x ATR(H1) distance from VWAP
    and a volume-climax M15 bar. Entry on M5 close back inside the climax range."""
    MIN_HOUR_UTC = 14
    MAX_HOUR_UTC = 19
    MIN_DIST_ATR = 2.5
    VOL_MULT = 2.5

    def __init__(self, *args, max_risk_atr=2.5, max_trades_per_day=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_risk_atr = max_risk_atr
        self.max_trades_per_day = max_trades_per_day
        self.pending = None
        self.last_m15_time = None
        self.last_m5_time = None
        self.trades_today = 0
        self.trades_date = None

    def _scan_m15(self):
        m15 = self.fetch_data(bars=120, timeframe=mt5.TIMEFRAME_M15)
        if m15 is None or len(m15) < 40:
            return
        bar = m15.iloc[-1]
        if bar['time'] == self.last_m15_time:
            return
        self.last_m15_time = bar['time']

        offset = broker_clock.offset
        t_utc = bar['time'] - pd.Timedelta(hours=offset)
        hour_now = int(t_utc.hour)
        if not (self.MIN_HOUR_UTC <= hour_now <= self.MAX_HOUR_UTC):
            return

        # Daily VWAP from M15 (reset at UTC midnight)
        t_local_utc = m15['time'] - pd.Timedelta(hours=offset)
        date = t_local_utc.dt.date
        tp = (m15['high'] + m15['low'] + m15['close']) / 3.0
        vol = m15['tick_volume']
        cum_v = vol.groupby(date).cumsum()
        cum_tpv = (tp * vol).groupby(date).cumsum()
        vwap = float((cum_tpv / np.maximum(cum_v, 1)).iloc[-1])
        close = float(bar['close'])

        # ATR(H1) proxy: use H1
        h1 = self.fetch_data(bars=20, timeframe=mt5.TIMEFRAME_H1)
        atr = compute_atr(h1, 14) if h1 is not None else 0.0
        if atr <= 0:
            return

        dist = close - vwap
        if abs(dist) < self.MIN_DIST_ATR * atr:
            return

        vr = volume_ratio(m15, len(m15) - 1)
        if vr < self.VOL_MULT:
            return

        # Climax detection
        bullish = dist < 0  # price below VWAP -> long reversion to VWAP
        if bullish:
            reversal = bar['close'] > bar['open']  # reversal candle
        else:
            reversal = bar['close'] < bar['open']
        if not reversal:
            return

        shared_state.count("vwap_setups_armed")
        self.pending = {
            "direction": BULL_SWEEP if bullish else BEAR_SWEEP,
            "extreme": float(bar['low'] if bullish else bar['high']),
            "climax_high": float(bar['high']),
            "climax_low": float(bar['low']),
            "vwap": vwap, "atr": atr, "dist": dist,
            "expires_at": time.time() + 1800, "attempts": 0,
            "bar_time": bar['time'],
        }
        logger.info(f"[{self.name}] ARMED VWAP reversion {'LONG' if bullish else 'SHORT'} "
                    f"(dist {dist:.2f}, vwap {vwap:.2f})")

    def _check_trigger(self):
        p = self.pending
        m5 = self.fetch_data(bars=20)
        if m5 is None or len(m5) < 5:
            return
        last_t = m5['time'].iloc[-1]
        if last_t == self.last_m5_time:
            return
        self.last_m5_time = last_t

        bullish = p['direction'] == BULL_SWEEP
        bar = m5.iloc[-1]
        close = float(bar['close'])

        # Invalidation: price reclaims the climax extreme against us
        if (bullish and close < p['extreme']) or ((not bullish) and close > p['extreme']):
            shared_state.count("vwap_failed")
            self._disarm()
            return

        # Trigger: M5 close back inside the climax M15 range
        if bullish:
            triggered = close > p['climax_high'] if False else close > p['climax_low']
            # we want close back above the climax low AND still above our SL
            triggered = close > p['climax_low'] and close > p['extreme']
        else:
            triggered = close < p['climax_high'] and close < p['extreme']

        if not triggered:
            return

        sl_raw = p['extreme']
        with mt5_lock:
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return
        entry = tick.ask if bullish else tick.bid
        risk = abs(entry - sl_raw)
        if risk > self.max_risk_atr * p['atr']:
            shared_state.count("vwap_risk_too_wide")
            self._disarm()
            return

        # RR: target VWAP, but use fixed RR fallback
        vwap_target_dist = abs(p['vwap'] - entry)
        rr_needed = vwap_target_dist / risk if risk > 0 else 0
        rr_used = max(1.5, min(3.0, rr_needed))

        logger.info(f"[{self.name}] M5 trigger for VWAP reversion (rr~{rr_used:.2f}).")
        shared_state.count("vwap_trigger")
        if self.execute_trade(p['direction'], sl_raw, tag="VWAPrev",
                              risk_pct=RISK_B_PCT, rr_override=rr_used):
            shared_state.count("vwap_trades")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                self._disarm()

    def _disarm(self):
        self.pending = None
        shared_state.update_vwap_setup(None)

    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            shared_state.update_agent_status(self.name, f"Daily cap ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        if self.pending is not None and time.time() > self.pending['expires_at']:
            shared_state.count("vwap_expired")
            self._disarm()

        with mt5_lock:
            busy = self.has_open_position()
        if busy and self.pending is None:
            shared_state.update_agent_status(self.name, "Idle (Position/cap)")
            return

        if self.pending is None:
            self._scan_m15()

        if self.pending is None:
            shared_state.update_vwap_setup(None)
            shared_state.update_agent_status(self.name, "Scanning VWAP distance")
            return

        p = self.pending
        shared_state.update_vwap_setup({
            "direction": p['direction'], "extreme": p['extreme'],
            "vwap": p['vwap'], "dist": p['dist'], "expires_at": p['expires_at'],
        })
        shared_state.update_agent_status(self.name, f"ARMED VWAP {p['direction']}")
        self._check_trigger()


# ==========================================================================
# 6. TRADE MANAGER — 3-STAGE EXIT + TRAILING + EOD FLAT
# ==========================================================================
class TradeManagerAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval,
                 be_trigger_r=1.0, be_buffer_points=80,
                 partial1_r=1.0, partial2_r=2.0,
                 trail_lookback_m15=3, flat_at_utc=FLAT_AT_UTC):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.be_trigger_r = be_trigger_r
        self.be_buffer_points = be_buffer_points
        self.partial1_r = partial1_r
        self.partial2_r = partial2_r
        self.trail_lookback_m15 = trail_lookback_m15
        self.flat_at_utc = flat_at_utc
        self._risks = {}
        self._originals = {}
        self._partial1_done = set()
        self._partial2_done = set()
        self._last_flat_check_date = None

    # ---------- helpers ----------
    def _fetch_m15_swing(self, is_buy):
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 6)
        if rates is None or len(rates) < 4:
            return None
        df = pd.DataFrame(rates).iloc[:-1]  # closed only
        if is_buy:
            return float(df['low'].iloc[-self.trail_lookback_m15:].min())
        return float(df['high'].iloc[-self.trail_lookback_m15:].max())

    def _partial_close(self, pos, volume, info, tag):
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        price = tick.bid if is_buy else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "position": pos.ticket,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": tag[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": ExecutionAgent._filling_mode(info),
        }
        with mt5_lock:
            r = mt5.order_send(req)
        if r is None:
            logger.warning(f"[{self.name}] partial_close None: {mt5.last_error()}")
            return False
        if r.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"[{self.name}] partial_close rejected: {r.retcode} {r.comment}")
            return False
        return True

    def _modify_sl(self, pos, new_sl, info):
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": round(new_sl, info.digits),
            "tp": pos.tp,
            "magic": pos.magic,
        }
        with mt5_lock:
            r = mt5.order_send(req)
        return bool(r and r.retcode == mt5.TRADE_RETCODE_DONE)

    # ---------- flat rule ----------
    def _force_flat_check(self, positions, info):
        n = now_utc()
        if not (n.hour > self.flat_at_utc or
                (n.hour == self.flat_at_utc and n.minute >= FLAT_AT_MINUTE)):
            return
        today_key = n.strftime('%Y-%m-%d')
        if self._last_flat_check_date == today_key:
            return
        self._last_flat_check_date = today_key
        for pos in positions:
            if pos.magic not in SYSTEM_MAGICS:
                continue
            logger.warning(f"[{self.name}] EOD flat: closing ticket {pos.ticket} (magic {pos.magic}).")
            self._partial_close(pos, pos.volume, info, "EOD_flat")
            shared_state.count("eod_flat_closes")

    # ---------- trailing ----------
    def _trail_runner(self, pos, info, risk):
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        profit = (pos.price_current - pos.price_open) if is_buy \
                 else (pos.price_open - pos.price_current)
        if profit < risk * self.partial2_r:
            return
        swing = self._fetch_m15_swing(is_buy)
        if swing is None:
            return
        buffer = self.be_buffer_points * info.point
        if is_buy:
            new_sl = swing - buffer
            if new_sl <= max(pos.sl, pos.price_open):
                return
        else:
            new_sl = swing + buffer
            if new_sl >= min(pos.sl, pos.price_open):
                return
        # Never trail into the current price
        if (is_buy and new_sl >= pos.price_current) or ((not is_buy) and new_sl <= pos.price_current):
            return
        if abs(pos.price_current - new_sl) < info.trade_stops_level * info.point:
            return
        if self._modify_sl(pos, new_sl, info):
            logger.info(f"[{self.name}] Trailed ticket {pos.ticket} to {new_sl:.2f}")

    # ---------- main ----------
    def step(self):
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            info = mt5.symbol_info(self.symbol)
        if positions is None or info is None:
            return

        mine = [p for p in positions if p.magic in SYSTEM_MAGICS]
        live = {p.ticket for p in mine}
        shared_state.update_agent_status(self.name, f"Managing {len(mine)} position(s)")

        # Force flat at EOD
        self._force_flat_check(mine, info)

        point, digits = info.point, info.digits
        stops_dist = info.trade_stops_level * point
        buffer = self.be_buffer_points * point

        for pos in mine:
            if pos.sl == 0.0 or pos.tp == 0.0:
                continue
            is_buy = pos.type == mt5.POSITION_TYPE_BUY

            # ----- original risk tracking -----
            if pos.ticket not in self._risks:
                on_loss_side = (pos.sl < pos.price_open) if is_buy else (pos.sl > pos.price_open)
                if on_loss_side:
                    self._risks[pos.ticket] = abs(pos.price_open - pos.sl)
                else:
                    # SL already at BE or better; approximate from TP assuming RR
                    self._risks[pos.ticket] = max(abs(pos.tp - pos.price_open) / 2.0, point * 100)
                self._originals[pos.ticket] = pos.volume

            risk = self._risks[pos.ticket]
            if risk <= 0:
                continue
            profit = (pos.price_current - pos.price_open) if is_buy \
                     else (pos.price_open - pos.price_current)

            # ----- PARTIAL 1 @ 1R -----
            if pos.ticket not in self._partial1_done and profit >= risk * self.partial1_r:
                pct = 0.40
                vol = math.floor((pos.volume * pct) / info.volume_step + 1e-9) * info.volume_step
                vol = round(round(vol / info.volume_step) * info.volume_step, 8)
                if vol >= info.volume_min:
                    if self._partial_close(pos, vol, info, "P1@1R"):
                        self._partial1_done.add(pos.ticket)
                        logger.info(f"[{self.name}] Ticket {pos.ticket} P1 40% @ 1R ({vol}).")
                        shared_state.count("partials_1R")

            # ----- PARTIAL 2 @ 2R -----
            if pos.ticket not in self._partial2_done and profit >= risk * self.partial2_r:
                pct = 0.50  # of the *remaining* (so ~30% of original)
                vol = math.floor((pos.volume * pct) / info.volume_step + 1e-9) * info.volume_step
                vol = round(round(vol / info.volume_step) * info.volume_step, 8)
                if vol >= info.volume_min:
                    if self._partial_close(pos, vol, info, "P2@2R"):
                        self._partial2_done.add(pos.ticket)
                        logger.info(f"[{self.name}] Ticket {pos.ticket} P2 30% @ 2R ({vol}).")
                        shared_state.count("partials_2R")

            # ----- BREAK-EVEN after 1R -----
            secured = (pos.sl >= pos.price_open) if is_buy else (pos.sl <= pos.price_open)
            if not secured and profit >= risk * self.be_trigger_r:
                new_sl = round(pos.price_open + buffer, digits) if is_buy \
                         else round(pos.price_open - buffer, digits)
                if not ((is_buy and new_sl >= pos.price_current) or
                        ((not is_buy) and new_sl <= pos.price_current)):
                    if abs(pos.price_current - new_sl) >= stops_dist:
                        if self._modify_sl(pos, new_sl, info):
                            logger.info(f"[{self.name}] Ticket {pos.ticket} -> BE ({new_sl}).")
                            shared_state.count("be_moves")

            # ----- TRAIL runner after 2R -----
            self._trail_runner(pos, info, risk)

        self._risks = {t: v for t, v in self._risks.items() if t in live}
        self._originals = {t: v for t, v in self._originals.items() if t in live}
        self._partial1_done = {t for t in self._partial1_done if t in live}
        self._partial2_done = {t for t in self._partial2_done if t in live}


# ==========================================================================
# 7. TELEMETRY AGENT
# ==========================================================================
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


# ==========================================================================
# 8. WEB DASHBOARD
# ==========================================================================
app = FastAPI(title="Wyckoff Bot Terminal")

@app.get("/api/state")
def get_system_state():
    return shared_state.export()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XAUUSD Multi-Agent Command Terminal v8</title>
<style>
:root{--bg:#0c0d0e;--card:#16181b;--line:#262a2f;--fg:#e1e3e6;--mut:#8b9098;--bull:#3fb950;--bear:#f85149;--warn:#d29922;--info:#58a6ff}
*{box-sizing:border-box}
body{margin:0;padding:20px;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.row{display:flex;justify-content:space-between;gap:12px;padding:3px 0;font-size:14px}
.row span:first-child{color:var(--mut)}
.bull{color:var(--bull)}.bear{color:var(--bear)}.warn{color:var(--warn)}.info{color:var(--info)}
.wide{grid-column:1/-1;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:500}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.badge{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}
.on{background:#238636;color:#fff}.off{background:#30363d;color:#8b949e}
.hot{background:#7d2029;color:#fff}
.empty{color:var(--mut);font-size:13px}
</style></head><body>
<h1>XAUUSD Multi-Agent Command Terminal v8</h1>
<div class="sub" id="conn">connecting...</div>
<div class="grid">
  <div class="card"><h2>Market</h2><div id="market"></div></div>
  <div class="card"><h2>Account</h2><div id="account"></div></div>
  <div class="card"><h2>H4 Macro (FVG)</h2><div id="macro"></div></div>
  <div class="card"><h2>Session</h2><div id="session"></div></div>
  <div class="card"><h2>Level sweeps (Lvl)</h2><div id="levels"></div></div>
  <div class="card"><h2>Failed breakout (FailBO)</h2><div id="failbo"></div></div>
  <div class="card"><h2>VWAP reversion</h2><div id="vwap"></div></div>
  <div class="card"><h2>Setup funnel</h2><div id="funnel"></div></div>
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
    row('Spread', `${num(m.spread,1)} pts`) +
    row('Trading hours', d.trading_hours_open ? '<span class="badge on">OPEN</span>' : '<span class="badge off">CLOSED</span>') +
    row('News blackout', d.news_blackout ? '<span class="badge hot">YES</span>' : 'no');

  $('account').innerHTML =
    row('Balance', '$' + num(a.balance)) +
    row('Equity', '$' + num(a.equity)) +
    row('Free margin', '$' + num(a.margin_free)) +
    row('Floating PnL', pnl(a.profit));

  const bcls = mc.bias === 'BULLISH' ? 'bull' : (mc.bias === 'BEARISH' ? 'bear' : '');
  const zones = mc.zones || [];
  $('macro').innerHTML =
    row('Bias', esc(mc.bias || '--'), bcls) +
    row('H4 in POI', mc.in_poi_h4 ? 'yes' : 'no') +
    row('Data', mc.fresh ? 'fresh' : 'STALE', mc.fresh ? '' : 'bear') +
    (zones.length
      ? zones.map((z,i)=>row('FVG '+(i+1), `${num(z.low)} - ${num(z.high)} (${z.age})`)).join('')
      : row('Zones', 'none valid', 'warn'));

  const sw = d.sweep;
  $('session').innerHTML =
    row('Killzone', s.in_killzone ? '<span class="badge on">OPEN</span>' : '<span class="badge off">CLOSED</span>') +
    row('Asian range', s.asian_high ? `${num(s.asian_low)} - ${num(s.asian_high)}` : 'none') +
    row('VWAP', s.vwap ? num(s.vwap) : '--') +
    row('H1 ATR', s.h1_atr ? num(s.h1_atr) : '--') +
    row('Vol ratio', num(s.vol_ratio) + 'x') +
    row('UTC offset', (s.broker_utc_offset >= 0 ? '+' : '') + s.broker_utc_offset + 'h') +
    row('Latched sweep', sw ? esc(sw.direction)+' ['+esc(sw.source)+']' : 'none', sw ? 'warn' : '');

  const lv = d.levels || {}, ls = d.level_setup;
  $('levels').innerHTML =
    (Object.keys(lv).length ? Object.keys(lv).slice(0,10).map(k => row(k, num(lv[k]))).join('')
                            : '<div class="empty">no levels yet</div>') +
    (ls ? row('Armed', esc(ls.direction)+' @ '+esc(ls.level_name), 'warn') +
          row('Score/grade', esc(ls.score)+' / '+esc(ls.grade)) +
          row('Why', esc(ls.breakdown)) +
          row('TTL', ls.expires_in_sec+'s')
        : row('Armed', 'none'));

  const fb = d.failbo_setup;
  $('failbo').innerHTML = fb
    ? row('Armed fade', esc(fb.direction)+' '+esc(fb.level_name), 'warn') +
      row('Level', num(fb.level)) +
      row('Extreme', num(fb.extreme)) +
      row('TTL', fb.expires_in_sec+'s')
    : '<div class="empty">none</div>';

  const vw = d.vwap_setup;
  $('vwap').innerHTML = vw
    ? row('Armed', esc(vw.direction), 'warn') +
      row('VWAP', num(vw.vwap)) +
      row('Dist from VWAP', num(vw.dist)) +
      row('TTL', vw.expires_in_sec+'s')
    : '<div class="empty">none</div>';

  const f = d.funnel || {};
  const fk = Object.keys(f).sort();
  $('funnel').innerHTML = fk.length ? fk.map(k => row(k.replace(/_/g,' '), f[k])).join('')
                                    : '<div class="empty">no setups yet</div>';

  const names = Object.keys(d.agents || {});
  $('agents').innerHTML =
    row('Priority', d.priority ? `<span class="badge on">${esc(d.priority.toUpperCase())}</span>` : 'NONE') +
    '<table><tr><th></th><th>Agent</th><th>Status</th><th>Loop</th></tr>' +
    names.map(n => {
      const g = d.agents[n];
      const col = g.stale ? 'var(--bear)' : 'var(--bull)';
      return `<tr><td><span class="dot" style="background:${col}"></span></td><td>${esc(n)}</td>` +
             `<td>${esc(g.status)}</td><td>${g.age === null ? '--' : g.age + 's'}</td></tr>`;
    }).join('') + '</table>';

  const t = d.trades || [];
  $('trades').innerHTML = t.length
    ? '<table><tr><th>Ticket</th><th>Tag</th><th>Side</th><th>Lots</th><th>Entry</th><th>Cur</th><th>SL</th><th>TP</th><th>PnL</th></tr>' +
      t.map(x => `<tr><td>${esc(x.ticket)}</td><td>${esc(x.agent)}</td>` +
        `<td class="${x.side === 'BUY' ? 'bull' : 'bear'}">${esc(x.side)}</td>` +
        `<td>${num(x.volume)}</td><td>${num(x.open)}</td><td>${num(x.current)}</td>` +
        `<td>${num(x.sl)}</td><td>${num(x.tp)}</td><td>${pnl(x.profit)}</td></tr>`).join('') + '</table>'
    : '<div class="empty">No open positions.</div>';

  const age = d.snapshot_age;
  $('conn').innerHTML = 'live · ' + new Date().toLocaleTimeString() +
    (age !== null && age > 15 ? ' · <span class="bear">snapshot '+age+'s old</span>' : '');
}

async function tick() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    if (!r.ok) throw new Error(r.status);
    render(await r.json());
  } catch (e) {
    $('conn').innerHTML = '<span class="bear">disconnected</span>';
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


# ==========================================================================
# 9. ORCHESTRATOR
# ==========================================================================
if __name__ == "__main__":
    missing = [n for n in REQUIRED_MT5_NAMES if not hasattr(mt5, n)]
    if missing:
        logger.error(f"MT5 package missing: {missing}")
        quit()

    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        quit()

    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Cannot select {SYMBOL}: {mt5.last_error()}")
        mt5.shutdown()
        quit()

    ALLOW_SHORTS = True
    clock = broker_clock

    macro = MacroAgent("Macro_H4", SYMBOL, mt5.TIMEFRAME_H4, sleep_interval=60)
    killzone = KillzoneAgent("Killzone_M15", SYMBOL, mt5.TIMEFRAME_M15, sleep_interval=15,
                             clock=clock, allow_shorts=ALLOW_SHORTS, sweep_ttl_sec=3600)
    swing = SwingAgent("Swing_H1", SYMBOL, mt5.TIMEFRAME_H1, sleep_interval=15,
                       magic=MAGIC_SWING, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=RISK_B_PCT,
                       rr=2.0, deviation=20, cooldown_sec=3600, min_sl_spread_mult=3.0,
                       max_spread_points=50.0, allow_shorts=ALLOW_SHORTS,
                       sweep_lookback=6, swing_n=30)
    levelsweep = LevelSweepAgent(
        "LevelSweep", SYMBOL, mt5.TIMEFRAME_M5, sleep_interval=10,
        magic=MAGIC_LEVEL, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=RISK_A_PCT,
        rr=2.5, deviation=20, cooldown_sec=900, min_sl_spread_mult=3.0,
        max_spread_points=50.0,
        allow_shorts=ALLOW_SHORTS, allow_counter_bias=False,
        min_score=5, a_score=7,
        pending_ttl_sec=2700, choch_lookback=6, max_risk_atr=3.0,
        max_trades_per_day=4,
    )
    failedbo = FailedBreakoutAgent(
        "FailedBO", SYMBOL, mt5.TIMEFRAME_M15, sleep_interval=20,
        magic=MAGIC_FAILBO, use_dynamic_lot=True, fixed_lot=0.05, risk_pct=RISK_B_PCT,
        rr=2.5, deviation=20, cooldown_sec=1200, min_sl_spread_mult=3.0,
        max_spread_points=50.0,
        allow_shorts=ALLOW_SHORTS, pending_ttl_sec=1800,
        max_risk_atr=2.5, max_trades_per_day=2,
    )
    vwaprev = VWAPReversionAgent(
        "VWAPRev", SYMBOL, mt5.TIMEFRAME_M5, sleep_interval=15,
        magic=MAGIC_VWAP, use_dynamic_lot=True, fixed_lot=0.05, risk_pct=RISK_B_PCT,
        rr=2.0, deviation=20, cooldown_sec=1200, min_sl_spread_mult=3.0,
        max_spread_points=50.0,
        max_risk_atr=2.5, max_trades_per_day=2,
    )
    continuation = ContinuationAgent(
        "Continuation_M15", SYMBOL, mt5.TIMEFRAME_M5, sleep_interval=15,
        magic=MAGIC_PYRAMID, use_dynamic_lot=True, fixed_lot=0.05, risk_pct=RISK_PYRAMID,
        rr=2.5, deviation=20, cooldown_sec=900, min_sl_spread_mult=3.0,
        max_spread_points=40.0
    )
    manager = TradeManagerAgent(
        "Manager", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=3,
        be_trigger_r=1.0, be_buffer_points=80,
        partial1_r=1.0, partial2_r=2.0,
        trail_lookback_m15=3,
    )
    telemetry = TelemetryAgent("Telemetry", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2)

    agents = [macro, killzone, swing, levelsweep, failedbo, vwaprev, continuation, manager, telemetry]
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
                    logger.error(f"!! [{a.name}] thread DEAD.")
                    warned.add(a.name)
                elif time.time() - a.last_beat > max(60, a.sleep_interval * 10):
                    logger.warning(f"!! [{a.name}] loop delay {time.time() - a.last_beat:.0f}s.")
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