# ==========================================================================
# multi_agent_portfolio_v9.py (Multi-Asset Edition) — patched
# ==========================================================================
import datetime
import json
import logging
import math
import os
import random
import threading
import time
import zlib
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ==========================================================================
# 0. CONFIGURATION & TARGET SYMBOLS
# ==========================================================================
TARGET_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCNH",
    "AUDUSD", "NZDUSD", "USDCAD", "USDSEK", "XAUUSD.sd"
]

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
    "history_deals_get",  # [FIX 2.7]
)

MAX_OPEN_POSITIONS = 2

# [FIX 5 / minor] Per-symbol tick age
MAX_TICK_AGE_SEC_DEFAULT = 15
def max_tick_age(symbol: str) -> int:
    s = symbol.upper()
    if "XAU" in s:
        return 30
    if "SEK" in s or "CNH" in s:
        return 30
    return MAX_TICK_AGE_SEC_DEFAULT

# --- Trading hours (UTC) ---
NO_ENTRY_BEFORE_UTC = 1
NO_ENTRY_AFTER_UTC  = 19
FLAT_AT_UTC         = 20
FLAT_AT_MINUTE      = 30

# --- Daily risk ---
DAILY_LOSS_CAP_PCT = 1.75               # realized (balance-based)
DAILY_FLOAT_DD_CAP_PCT = 2.25           # [FIX 2.2] floating gate
SYMBOL_DAILY_LOSS_CAP_PCT = 0.50        # [FIX 2.7]

# --- Correlation guard ---
# [FIX 2.1] Cap concurrent same-direction USD-linked exposure
MAX_USD_SAME_SIDE = 3
USD_LINKED_FAMILY = {
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY", "USDCNH", "USDSEK",
}

# --- Global per-hour trade throttle ---
# [FIX 2.3]
MAX_GLOBAL_TRADES_PER_HOUR = 6

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

# [FIX 4.6] Structured event log
_evt_logger = logging.getLogger("WyckoffBot.events")
_evt_logger.setLevel(logging.INFO)
_evt_logger.propagate = False

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

    _evt_fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "events.jsonl"),
        maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    _evt_fh.setFormatter(logging.Formatter('%(message)s'))
    _evt_logger.addHandler(_evt_fh)


def log_event(event, **fields):
    # [FIX 4.6] One JSON line per decision for downstream analysis.
    payload = {"ts": time.time(), "event": event, **fields}
    try:
        _evt_logger.info(json.dumps(payload, default=str))
    except Exception:
        logger.warning(f"log_event failed: {event}")


# ==========================================================================
# 0.5 BROKER CLOCK
# ==========================================================================
class BrokerClock:
    """
    Fixed broker UTC offset. No verification — the offset is a known
    property of the broker and does not change at runtime.
    """
    def __init__(self, offset_hours):
        self.offset = offset_hours
        self.verified = True        # treat as authoritative from the start

    def refresh(self, symbol):
        # No-op: nothing to refresh.
        return

    def tick_age(self, tick):
        if tick is None or not tick.time:
            return None
        return time.time() + self.offset * 3600 - tick.time

    def warn_if_unverified(self, seconds=180):
        # No-op: never unverified.
        return


broker_clock = BrokerClock(BROKER_UTC_OFFSET_FALLBACK)
logger.info(f"Broker clock: using fixed UTC offset {broker_clock.offset:+d}h.")


broker_clock = BrokerClock(BROKER_UTC_OFFSET_FALLBACK)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def now_broker():
    # [FIX 3.9] Broker-time helper for EOD flat and anything broker-wall-clock.
    return now_utc() + datetime.timedelta(hours=broker_clock.offset)


def in_trading_hours():
    # Config values NO_ENTRY_* are UTC-based by design.
    h = now_utc().hour
    return NO_ENTRY_BEFORE_UTC <= h < NO_ENTRY_AFTER_UTC


def is_flat_time():
    # [FIX 3.10] Removed — TradeManager has its own broker-time check.
    return 


def news_blackout():
    n = now_utc()
    m = n.hour * 60 + n.minute
    # NFP first Friday
    if n.weekday() == 4 and n.day <= 7 and (13*60+15) <= m <= (14*60+15):
        return True
    # (Comment corrected for clarity; FOMC is Wed ~19:00 UTC.)
    if 8 <= n.day <= 15 and n.weekday() in (1, 2) and (13*60+15) <= m <= (14*60+15):
        return True
    if (14*60+55) <= m <= (15*60+15):
        return True
    return False


# ==========================================================================
# 0.55 GLOBAL KILL SWITCH  [FIX 4.8]
# ==========================================================================
class TradingEnabled:
    def __init__(self):
        self._enabled = True
        self._lock = threading.Lock()

    def is_enabled(self):
        with self._lock:
            return self._enabled

    def set(self, v):
        with self._lock:
            self._enabled = bool(v)


trading_enabled = TradingEnabled()


# ==========================================================================
# 0.56 GLOBAL TRADE RATE LIMITER  [FIX 2.3]
# ==========================================================================
class HourlyTradeLimiter:
    def __init__(self, max_per_hour):
        self.max = max_per_hour
        self._lock = threading.Lock()
        self._events = []

    def can_acquire(self):
        now = time.time()
        cutoff = now - 3600
        with self._lock:
            self._events = [t for t in self._events if t > cutoff]
            return len(self._events) < self.max

    def record(self):
        with self._lock:
            self._events.append(time.time())


hourly_limiter = HourlyTradeLimiter(MAX_GLOBAL_TRADES_PER_HOUR)


# ==========================================================================
# 0.6 DAILY RISK BREAKER  [FIX 1.1, 1.2, 2.2]
# ==========================================================================
class DailyRiskState:
    def __init__(self, cap_pct, state_file, floating_cap_pct=None):
        self.cap_pct = cap_pct
        self.floating_cap_pct = floating_cap_pct if floating_cap_pct is not None else cap_pct
        self.state_file = state_file
        self._lock = threading.Lock()
        self._date = None
        self._start_balance = None
        self._day_anchor_equity = None
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    d = json.load(f)
                self._date = d.get("date")
                self._start_balance = d.get("start_balance")
                self._day_anchor_equity = d.get("day_anchor_equity")
                if self._date:
                    logger.info(
                        f"Daily risk state loaded: {self._date} "
                        f"start_balance={self._start_balance} "
                        f"anchor_eq={self._day_anchor_equity}"
                    )
        except Exception as e:
            logger.warning(f"Could not load daily risk state: {e}")

    def _save(self):
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "date": self._date,
                    "start_balance": self._start_balance,
                    "day_anchor_equity": self._day_anchor_equity,
                }, f)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.warning(f"Could not save daily risk state: {e}")

    def check(self, acct, broker_offset_hours=0):
        # [FIX 1.1] Day boundary computed in broker time.
        if acct is None:
            return True
        today = time.strftime(
            '%Y-%m-%d',
            time.gmtime(time.time() + broker_offset_hours * 3600),
        )
        with self._lock:
            if self._date != today:
                self._date = today
                self._start_balance = acct.balance
                self._day_anchor_equity = acct.equity
                self._save()
                return False

            # Recover from a mid-day restart with a corrupted/blank file.
            if self._start_balance is None or self._start_balance <= 0:
                self._start_balance = acct.balance
                self._day_anchor_equity = acct.equity
                self._save()
                return False

            realized_dd = (self._start_balance - acct.balance) / self._start_balance
            if realized_dd * 100 >= self.cap_pct:
                return True

            if self._day_anchor_equity and self._day_anchor_equity > 0:
                float_dd = (self._day_anchor_equity - acct.equity) / self._day_anchor_equity
                if float_dd * 100 >= self.floating_cap_pct:
                    return True

            return False


daily_risk = DailyRiskState(
    DAILY_LOSS_CAP_PCT,
    os.path.join(LOG_DIR, "daily_risk.json"),
    floating_cap_pct=DAILY_FLOAT_DD_CAP_PCT,
)


# ==========================================================================
# 0.65 PER-SYMBOL DAILY PNL  [FIX 2.7]
# ==========================================================================
def symbol_realized_pnl_today(symbol, magics, broker_offset_hours=0):
    now_ts = time.time()
    broker_day = time.strftime(
        '%Y-%m-%d', time.gmtime(now_ts + broker_offset_hours * 3600))
    day_start_utc = datetime.datetime.strptime(broker_day, '%Y-%m-%d') \
        .replace(tzinfo=datetime.timezone.utc) \
        - datetime.timedelta(hours=broker_offset_hours)
    from_ts = int(day_start_utc.timestamp())
    to_ts = int(now_ts) + 1
    with mt5_lock:
        deals = mt5.history_deals_get(from_ts, to_ts)
    if deals is None:
        return 0.0
    total = 0.0
    for d in deals:
        if d.symbol != symbol:
            continue
        if d.magic not in magics:
            continue
        total += (d.profit or 0.0) + (d.swap or 0.0) + (d.commission or 0.0)
    return total


# ==========================================================================
# 0.66 CORRELATION GUARD  [FIX 2.1]
# ==========================================================================
def usd_side_of(symbol, is_buy):
    """+1 = long USD, -1 = short USD, 0 = neither."""
    s = symbol.split(".")[0]
    if s.startswith("USD"):
        return 1 if is_buy else -1
    if s.endswith("USD"):
        return -1 if is_buy else 1
    return 0


def usd_net_counts(all_magics):
    with mt5_lock:
        positions = mt5.positions_get() or []
    longs = shorts = 0
    for p in positions:
        if p.magic not in all_magics:
            continue
        base = p.symbol.split(".")[0]
        if base not in USD_LINKED_FAMILY:
            continue
        is_buy = p.type == mt5.POSITION_TYPE_BUY
        if usd_side_of(p.symbol, is_buy) > 0:
            longs += 1
        else:
            shorts += 1
    return longs, shorts


# ==========================================================================
# 0.7 STABLE MAGIC BASES  [FIX 4.1]
# ==========================================================================
MAGIC_SLOT = {"SWING": 1, "LEVEL": 2, "PYRAMID": 3, "FAILBO": 4, "VWAP": 5, "KZ_SWEEP": 6}
_MAGIC_FILE = os.path.join(LOG_DIR, "magic_bases.json")
_magic_base_cache = {}


def _load_magic_bases():
    global _magic_base_cache
    try:
        if os.path.exists(_MAGIC_FILE):
            with open(_MAGIC_FILE) as f:
                _magic_base_cache = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load magic bases: {e}")
        _magic_base_cache = {}


def _save_magic_bases():
    try:
        tmp = _MAGIC_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_magic_base_cache, f, indent=2)
        os.replace(tmp, _MAGIC_FILE)
    except Exception as e:
        logger.warning(f"Could not save magic bases: {e}")


def get_or_create_symbol_base(symbol):
    if symbol in _magic_base_cache:
        return int(_magic_base_cache[symbol])
    used = {int(v) for v in _magic_base_cache.values()}
    base = 100000
    while base in used:
        base += 100000
    _magic_base_cache[symbol] = base
    _save_magic_bases()
    logger.info(f"Assigned stable magic base {base} to {symbol}.")
    return base


# ==========================================================================
# 1. ISOLATED SYMBOL STATE & TELEMETRY
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

        # [FIX 3.8] Shared H1 ATR cache (per symbol).
        self._atr_cache = 0.0
        self._atr_cache_ts = 0.0

        # [FIX 2.7] Shared daily PnL cache (per symbol).
        self._pnl_cache = 0.0
        self._pnl_cache_ts = 0.0

    # -- shared ATR -- [FIX 3.8]
    def get_h1_atr(self, symbol, ttl=300):
        now = time.time()
        if now - self._atr_cache_ts < ttl and self._atr_cache > 0:
            return self._atr_cache
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
        if rates is None or len(rates) < 15:
            return self._atr_cache or 0.0
        df = pd.DataFrame(rates).iloc[:-1]
        self._atr_cache = compute_atr(df, 14)
        self._atr_cache_ts = now
        return self._atr_cache

    # -- shared symbol PnL -- [FIX 2.7]
    def get_symbol_daily_pnl(self, symbol, magics, ttl=30):
        now = time.time()
        if now - self._pnl_cache_ts < ttl:
            return self._pnl_cache
        val = symbol_realized_pnl_today(symbol, magics, broker_clock.offset)
        self._pnl_cache = val
        self._pnl_cache_ts = now
        return val

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

    def clear_sweep_if(self, bar_time):
        # [FIX 1.6] Atomic compare-and-clear to avoid stomping a newer latch.
        with self.state_lock:
            if self._sweep is not None and self._sweep.bar_time == bar_time:
                self._sweep = None
                return True
            return False

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
                "trading_enabled": trading_enabled.is_enabled(),  # [FIX 4.8]
            }


symbol_states = {}
ALL_SYSTEM_MAGICS = set()  # [FIX 2.1] populated at startup


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
    def __init__(self, name, symbol, timeframe, sleep_interval, state):
        super().__init__(name=f"{name}_{symbol}", daemon=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.sleep_interval = sleep_interval
        self.state = state
        self.base_name = name
        self._stop_event = threading.Event()
        self.last_beat = time.time()
        self.state.update_agent_status(self.base_name, "Initializing...")

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
        time.sleep(random.uniform(0.1, 3.0))
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception as e:
                logger.error(f"[{self.name}] Error in step(): {e}", exc_info=True)
            # [FIX 4.3] Beat only after a full cycle completes.
            self.last_beat = time.time()
            self.state.beat(self.base_name, self.sleep_interval)
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
            self.state.update_agent_status(self.base_name, "Awaiting Data...")
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

        self.state.update_macro(bias, zones, in_poi_h4)
        self.state.update_agent_status(
            self.base_name, f"Bias: {bias} | {len(zones)} FVG zone(s) | H4 in POI: {in_poi_h4}")


# ==========================================================================
# 4. KILLZONE AGENT (M15)
# ==========================================================================
class KillzoneAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, state, clock,
                 asian_end_hour=6, killzones=((7, 10), (12, 15)), vol_mult=1.5,
                 allow_shorts=True, sweep_ttl_sec=3600):
        super().__init__(name, symbol, timeframe, sleep_interval, state)
        self.clock = clock
        self.asian_end_hour = asian_end_hour
        self.killzones = killzones
        self.vol_mult = vol_mult
        self.allow_shorts = allow_shorts
        self.sweep_ttl_sec = sweep_ttl_sec
        self.last_bar_time = None
        self.asian_high = 0.0
        self.asian_low = 0.0
        self.asian_date = None
        self.min_asian_bars = 20 if asian_end_hour >= 5 else 10

    def step(self):
        self.clock.refresh(self.symbol)
        offset = self.clock.offset

        bias, has_zone, fresh = self.state.get_macro()
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
        asian = df[(date == today) & (hour < self.asian_end_hour)]
        if len(asian) >= self.min_asian_bars:
            self.asian_high = float(asian['high'].max())
            self.asian_low = float(asian['low'].min())
            self.asian_date = today
        elif self.asian_date != today:
            self.asian_high = self.asian_low = 0.0

        hour_now = int(hour.iloc[-1])
        in_kz = any(a <= hour_now <= b for a, b in self.killzones)
        vol_ratio = volume_ratio(df, n - 1)
        h1 = self.fetch_data(bars=20, timeframe=mt5.TIMEFRAME_H1)
        h1_atr = compute_atr(h1, 14) if h1 is not None else 0.0

        self.state.update_session_telemetry(
            asian_high=self.asian_high, asian_low=self.asian_low,
            vol_ratio=vol_ratio, in_killzone=in_kz, vwap=vwap,
            broker_utc_offset=offset, h1_atr=h1_atr)

        if bar['time'] == self.last_bar_time:
            return
        self.last_bar_time = bar['time']

        if bias not in ("BULLISH", "BEARISH") or not fresh:
            self.state.update_agent_status(self.base_name, "Idle (Macro data stale)")
            return
        if not has_zone:
            self.state.update_agent_status(self.base_name, "Idle (No valid H4 zone)")
            return
        if not in_kz:
            self.state.update_agent_status(
                self.base_name, f"Outside KZ (Asian {self.asian_low:.5f}-{self.asian_high:.5f})")
            return
        if self.asian_high <= 0 or self.asian_low <= 0:
            self.state.update_agent_status(self.base_name, "Idle (No Asian range today)")
            return
        if self.state.get_sweep() is not None:
            self.state.update_agent_status(self.base_name, "Sweep latched")
            return

        self.state.count("kz_bars_scanned_in_window")
        bullish = bias == "BULLISH"
        climax = vol_ratio > self.vol_mult

        prior = df.iloc[:-1]
        prior_after = prior[(date.iloc[:-1] == today) & (hour.iloc[:-1] >= self.asian_end_hour)]

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
            self.state.update_agent_status(
                self.base_name, f"Scanning KZ | Vol {vol_ratio:.2f}x")
            return

        self.state.count("kz_sweep_pattern")
        if already_taken:
            self.state.update_agent_status(self.base_name, "Level already taken today")
            return

        zone = self.state.zone_touched(float(bar['low']), float(bar['high']))
        if zone is None:
            self.state.update_agent_status(self.base_name, "Sweep outside H4 FVG")
            return

        self.state.count("kz_fresh_and_at_h4_zone")
        bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
        self.state.set_sweep(Sweep(
            direction=direction, level=level, extreme=extreme,
            bar_time=bar['time'], close_time=bar['time'] + bar_delta,
            expires_at=time.time() + self.sweep_ttl_sec, source="ASIAN",
        ))
        logger.info(f"[{self.name}] {direction} latched at {bar['time']} "
                    f"(level {level:.5f}, extreme {extreme:.5f}, "
                    f"H4 FVG {zone['low']:.5f}-{zone['high']:.5f})")
        log_event("kz_sweep_latched", symbol=self.symbol, direction=direction,
                  level=level, extreme=extreme, source="ASIAN")
        self.state.update_agent_status(self.base_name, f"LATCHED: {direction}")


# ==========================================================================
# 5. EXECUTION BASE CLASS
# ==========================================================================
class ExecutionAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, state, magic, system_magics,
                 use_dynamic_lot, fixed_lot, risk_pct,
                 rr=2.0, deviation=20, cooldown_sec=300,
                 min_sl_spread_mult=3.0, max_spread_points=20.0):
        super().__init__(name, symbol, timeframe, sleep_interval, state)
        self.magic = magic
        self.system_magics = system_magics
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

    # ---------- risk ----------
    def _daily_loss_breaker(self):
        with mt5_lock:
            acct = mt5.account_info()
        # [FIX 1.1] Pass broker offset so day boundary matches broker day.
        return daily_risk.check(acct, broker_clock.offset)

    def _h1_atr(self):
        # [FIX 3.8] Delegate to shared per-symbol cache.
        return self.state.get_h1_atr(self.symbol)

    # ---------- position checks ----------
    def cannot_open_new(self):
        # [FIX 1.4] All MT5 access under lock.
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return True
        mine = [p for p in positions if p.magic in self.system_magics]
        if any(p.magic == self.magic for p in mine):
            return True
        return len(mine) >= MAX_OPEN_POSITIONS

    @staticmethod
    def _filling_mode(info):
        # [minor] Prefer FOK when supported, then IOC, then RETURN.
        if info.filling_mode & SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if info.filling_mode & SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _skip(self, reason):
        logger.warning(f"[{self.name}] Trade skipped: {reason}")
        log_event("trade_skipped", symbol=self.symbol, agent=self.base_name, reason=reason)
        return False

    def calculate_lot_size(self, info, entry, sl, risk_pct=None):
        rp = self.risk_pct if risk_pct is None else risk_pct
        step = info.volume_step
        vmin, vmax = info.volume_min, info.volume_max

        if not self.use_dynamic_lot:
            lot = math.floor(self.fixed_lot / step + 1e-9) * step
            lot = max(vmin, min(lot, vmax))
            return round(round(lot / step) * step, 8)

        with mt5_lock:
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

        if not trading_enabled.is_enabled():          # [FIX 4.8]
            return self._skip("global halt active")
        if not in_trading_hours():
            return self._skip("outside trading hours (no-entry window)")
        if news_blackout():
            return self._skip("news blackout window")

        # [FIX 2.3] Global per-hour throttle
        if not hourly_limiter.can_acquire():
            return self._skip("global hourly trade cap reached")

        with mt5_lock:
            if self._daily_loss_breaker():
                return self._skip("daily loss breaker active")

            # [FIX 2.7] Per-symbol daily loss cap
            pnl_today = self.state.get_symbol_daily_pnl(self.symbol, self.system_magics)
            with mt5_lock:
                acct = mt5.account_info()
            if acct is not None:
                cap_amt = acct.balance * (SYMBOL_DAILY_LOSS_CAP_PCT / 100.0)
                if pnl_today <= -cap_amt:
                    return self._skip(
                        f"per-symbol daily loss cap hit ({pnl_today:.2f} <= -{cap_amt:.2f})")

            # [FIX 1.5] Cap check now uses _skip so it's logged/telemetered.
            if self.cannot_open_new():
                return self._skip("position cap reached")

            # [FIX 2.1] Correlation guard
            longs, shorts = usd_net_counts(ALL_SYSTEM_MAGICS)
            base = self.symbol.split(".")[0]
            if base in USD_LINKED_FAMILY:
                wants_long_usd = usd_side_of(self.symbol, is_buy) > 0
                if wants_long_usd and longs >= MAX_USD_SAME_SIDE:
                    return self._skip(f"USD-long bucket cap ({longs} open)")
                if (not wants_long_usd) and shorts >= MAX_USD_SAME_SIDE:
                    return self._skip(f"USD-short bucket cap ({shorts} open)")

            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
            if info is None or tick is None:
                return self._skip(f"no symbol/tick ({mt5.last_error()})")

            terminal = mt5.terminal_info()
            if terminal is None or not terminal.connected:
                return self._skip("terminal disconnected")

            # [minor] Per-symbol tick-age limit
            age = broker_clock.tick_age(tick)
            if age is not None and abs(age) > max_tick_age(self.symbol):
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
                return self._skip(f"SL {risk:.5f} < min {min_dist:.5f}")

            if max_risk_override is not None and risk > max_risk_override:
                return self._skip(f"risk {risk:.5f} > cap {max_risk_override:.5f}")

            rr_used = rr_override if rr_override is not None else self.rr
            atr = self._h1_atr()
            target_dist = risk * rr_used
            if atr > 0:
                target_dist = max(target_dist, 0.6 * atr)
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
                "comment": (tag or self.base_name)[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(info),
            }
            result = mt5.order_send(request)

        if result is None:
            logger.warning(f"[{self.name}] order_send None: {mt5.last_error()}")
            log_event("order_send_none", symbol=self.symbol, agent=self.base_name)
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"[{self.name}] Rejected: {result.retcode} {result.comment}")
            log_event("order_rejected", symbol=self.symbol, agent=self.base_name,
                      retcode=result.retcode, comment=result.comment)
            return False

        # [FIX 2.3] Record successful execution in the global throttle.
        hourly_limiter.record()

        logger.info(f"[{self.name}] Filled {'BUY' if is_buy else 'SELL'} {lot} @ "
                    f"{result.price}. SL {request['sl']}, TP {request['tp']} [{request['comment']}]")
        log_event("trade_opened", symbol=self.symbol, agent=self.base_name,
                  side="BUY" if is_buy else "SELL", volume=lot, price=result.price,
                  sl=request["sl"], tp=request["tp"], tag=request["comment"],
                  magic=self.magic)
        return True


# ==========================================================================
# 5.0 KILLZONE SWEEP AGENT
# ==========================================================================
class KillzoneSweepAgent(ExecutionAgent):
    def __init__(self, *args, max_risk_atr=2.0, retry_throttle_sec=90, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_risk_atr = max_risk_atr
        self.retry_throttle_sec = retry_throttle_sec
        self._handled_bar_time = None
        self._last_attempt_ts = 0.0

    def step(self):
        if time.time() < self.cooldown_until:
            self.state.update_agent_status(
                self.base_name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        prio = self.state.get_priority()
        if prio in ("level", "failbo"):
            self.state.update_agent_status(self.base_name, f"Yielding ({prio} has priority)")
            return

        sweep = self.state.get_sweep()
        if sweep is None:
            self.state.update_agent_status(self.base_name, "Idle (no sweep latched)")
            return
        if sweep.bar_time == self._handled_bar_time:
            self.state.update_agent_status(self.base_name, "Sweep already handled")
            return
        if time.time() - self._last_attempt_ts < self.retry_throttle_sec:
            self.state.update_agent_status(self.base_name, "Retry throttle")
            return

        if self.cannot_open_new():   # [FIX 1.4] now lock-safe internally
            self.state.update_agent_status(self.base_name, "Idle (position/cap)")
            return

        with mt5_lock:
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return

        atr = self._h1_atr()
        if atr <= 0:
            return

        is_buy = sweep.direction == BULL_SWEEP
        entry = tick.ask if is_buy else tick.bid
        risk = abs(entry - sweep.extreme)
        if risk <= 0 or risk > self.max_risk_atr * atr:
            self.state.update_agent_status(
                self.base_name, f"Sweep risk {risk:.5f} > {self.max_risk_atr} ATR; discarding")
            # [FIX 1.6] Clear by token so a concurrent latch isn't lost.
            self.state.clear_sweep_if(sweep.bar_time)
            self._handled_bar_time = sweep.bar_time
            return

        logger.info(f"[{self.name}] Executing latched {sweep.direction} "
                    f"[{sweep.source}] level {sweep.level:.5f} extreme {sweep.extreme:.5f}")
        self.state.count("kz_sweep_attempts")
        self._last_attempt_ts = time.time()

        tag = f"KZ:{sweep.source}"
        if self.execute_trade(sweep.direction, sweep.extreme, tag=tag, rr_override=2.5):
            self.state.count("kz_sweep_trades")
            # [FIX 1.6] Clear only if it's still the same sweep.
            self.state.clear_sweep_if(sweep.bar_time)
            self._handled_bar_time = sweep.bar_time
            self.cooldown_until = time.time() + self.cooldown_sec
        else:
            self.state.update_agent_status(self.base_name, "Entry deferred (retry throttled)")


# ==========================================================================
# 5.1 SWING AGENT (H1)
# ==========================================================================
class SwingAgent(ExecutionAgent):
    def __init__(self, *args, allow_shorts=False, sweep_lookback=6, swing_n=30, vol_mult=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.sweep_lookback = sweep_lookback
        self.swing_n = swing_n
        self.vol_mult = vol_mult

    def step(self):
        if time.time() < self.cooldown_until:
            self.state.update_agent_status(self.base_name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return
        prio = self.state.get_priority()
        if prio in ("level", "failbo"):
            self.state.update_agent_status(self.base_name, f"Yielding ({prio} has priority)")
            return

        bias, has_zone, fresh = self.state.get_macro()
        if not (fresh and has_zone) or bias not in ("BULLISH", "BEARISH"):
            self.state.update_agent_status(self.base_name, "Idle (No valid H4 zone)")
            return
        if bias == "BEARISH" and not self.allow_shorts:
            self.state.update_agent_status(self.base_name, "Idle (Shorts disabled)")
            return

        if self.cannot_open_new():   # [FIX 1.4]
            self.state.update_agent_status(self.base_name, "Idle (Position open)")
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
        self.state.update_agent_status(self.base_name, f"Hunting H1 sweep+FVG ({bias})")

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
        self.state.count("swing_h1_fvg")

        found = None
        # [FIX 3.3] Guard the scan range against the sweep detector's window.
        j_lo = max(self.swing_n + 3, n - 3 - self.sweep_lookback)
        for j in range(n - 3, j_lo - 1, -1):
            hit, level, extreme = detect_sweep(df, j, bullish, swing_n=self.swing_n, vol_mult=self.vol_mult)
            if not hit:
                continue
            bar_j = df.iloc[j]
            zone = self.state.zone_touched(float(bar_j['low']), float(bar_j['high']))
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

        self.state.count("swing_h1_fvg_with_sweep")
        j, level, extreme, zone = found
        logger.info(f"[{self.name}] H1 sweep {df['time'].iloc[j]} @ H4 {zone['low']:.5f}-{zone['high']:.5f} + FVG.")
        if self.execute_trade(direction, sl_raw, tag=f"Swing:H1", rr_override=2.0):
            self.state.count("swing_h1_trades")
            self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================================================
# 5.2 CONTINUATION AGENT (M15 pyramid, gated on parent 1.2R)
# ==========================================================================
class ContinuationAgent(ExecutionAgent):
    PARENT_MIN_PROFIT_R = 1.2

    def __init__(self, *args, parent_magic=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_risk = {}
        self.parent_magic = parent_magic

    def _parent_risk_for(self, p, is_buy):
        if p.ticket in self._parent_risk:
            return self._parent_risk[p.ticket]
        on_loss_side = (p.sl < p.price_open) if is_buy else (p.sl > p.price_open)
        if on_loss_side and abs(p.price_open - p.sl) > 0:
            risk = abs(p.price_open - p.sl)
        elif p.tp and abs(p.tp - p.price_open) > 0:
            # [FIX 1.9] Prefer a name-tagged RR if the comment reveals it; else
            # fall back to a conservative 2.5 (matches LevelSweep defaults of 2.0-3.0).
            rr_from_tag = 2.5
            tag = (p.comment or "").upper()
            if tag.startswith("LVL:"):
                # "Lvl:PDH:A" → grade in the last colon-separated token
                try:
                    grade = tag.split(":")[-1]
                    rr_from_tag = 3.0 if grade == "A" else 2.0
                except Exception:
                    pass
            risk = abs(p.tp - p.price_open) / rr_from_tag
        else:
            return None
        if risk <= 0:
            return None
        self._parent_risk[p.ticket] = risk
        return risk

    def step(self):
        if time.time() < self.cooldown_until:
            self.state.update_agent_status(self.base_name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        parent = None
        parent_risk = 0.0
        live_tickets = set()
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            self.state.update_agent_status(self.base_name, "Idle (positions unavailable)")
            return

        for p in positions:
            live_tickets.add(p.ticket)
            if p.magic != self.parent_magic:
                continue
            is_buy = p.type == mt5.POSITION_TYPE_BUY
            risk = self._parent_risk_for(p, is_buy)
            if risk is None or risk <= 0:
                continue
            is_be = (p.sl >= p.price_open) if is_buy else (p.sl <= p.price_open)
            if not is_be:
                continue
            profit = (p.price_current - p.price_open) if is_buy \
                     else (p.price_open - p.price_current)
            if profit >= risk * self.PARENT_MIN_PROFIT_R:
                parent = p
                parent_risk = risk
                break

        self._parent_risk = {t: v for t, v in self._parent_risk.items() if t in live_tickets}

        if parent is None:
            self.state.update_agent_status(self.base_name, f"Idle (no parent >= {self.PARENT_MIN_PROFIT_R}R)")
            return

        if self.cannot_open_new():   # [FIX 1.4]
            self.state.update_agent_status(self.base_name, "Idle (Pyramid already open)")
            return

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
            sl_raw = max(structural_sl, parent.sl)
        else:
            has_fvg = c1['low'] > c3['high'] and c2['close'] < c2['open'] and c3['close'] < c1['low']
            structural_sl = float(c1['high'])
            sl_raw = min(structural_sl, parent.sl)

        if not has_fvg:
            self.state.update_agent_status(self.base_name, f"Waiting M15 FVG for {direction}")
            return

        with mt5_lock:
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
        if info is None or tick is None:
            return
        entry = tick.ask if is_buy else tick.bid

        # [FIX 3.4] Allow a tolerance of 25% of FVG size so gap-ins don't reject.
        if is_buy:
            fvg_top, fvg_bottom = float(c3['low']), float(c1['high'])
            fvg_size = max(fvg_top - fvg_bottom, info.point)
            tol = 0.25 * fvg_size
            if not (fvg_bottom - tol <= entry <= fvg_top + tol):
                self.state.update_agent_status(self.base_name, "Waiting for FVG retest")
                return
        else:
            fvg_top, fvg_bottom = float(c1['low']), float(c3['high'])
            fvg_size = max(fvg_top - fvg_bottom, info.point)
            tol = 0.25 * fvg_size
            if not (fvg_bottom - tol <= entry <= fvg_top + tol):
                self.state.update_agent_status(self.base_name, "Waiting for FVG retest")
                return

        pyramid_risk = abs(entry - sl_raw)
        max_allowed = parent_risk * 1.5
        if pyramid_risk > max_allowed:
            logger.info(f"[{self.name}] Pyramid SL {pyramid_risk:.5f} > "
                        f"1.5x parent risk {max_allowed:.5f}; skipped.")
            self.state.count("pyramid_risk_rejected")
            return

        logger.info(f"[{self.name}] M15 FVG for pyramid (parent ticket {parent.ticket}).")
        self.state.count("pyramid_m15_triggered")
        if self.execute_trade(direction, sl_raw, tag="Pyr:M15",
                              risk_pct=RISK_PYRAMID, rr_override=2.5,
                              max_risk_override=max_allowed):
            self.state.count("pyramid_trades")
            self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================================================
# 5.5 LEVEL SWEEP AGENT (PDH/PDL + London H/L + Round numbers)
# ==========================================================================
class LevelSweepAgent(ExecutionAgent):
    LOW_LEVELS = ("PDL", "LDNL")
    HIGH_LEVELS = ("PDH", "LDNH")
    HTF_REFRESH_SEC = 600

    def __init__(self, *args, allow_shorts=True, allow_counter_bias=False,
                 min_score=5, a_score=7, pending_ttl_sec=2700, choch_lookback=6,
                 max_risk_atr=3.0, max_trades_per_day=4, london_hours=(7, 12),
                 killzones=((7, 10), (12, 15)), round_number_step=0.0050,
                 vol_mult_climax=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.allow_counter_bias = allow_counter_bias
        self.min_score = min_score
        self.a_score = a_score
        self.pending_ttl_sec = pending_ttl_sec
        self.choch_lookback = choch_lookback
        self.max_risk_atr = max_risk_atr
        self.max_trades_per_day = max_trades_per_day
        self.london_hours = london_hours
        self.killzones = killzones
        self.round_number_step = round_number_step
        self.vol_mult_climax = vol_mult_climax

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
            # [FIX 1.7] D1 open anchors the fixed round-number grid.
            htf["DO"] = float(d1['open'].iloc[-1])
        if w1 is not None and len(w1) >= 1:
            htf["WO"] = float(w1['open'].iloc[-1])
        self._htf = htf
        self._htf_ts = now if htf else now - self.HTF_REFRESH_SEC + 30

    def _london_levels(self, df, offset):
        t = df['time'] - pd.Timedelta(hours=offset)
        hour, date = t.dt.hour, t.dt.date
        today = date.iloc[-1]
        sess = df[(date == today) & (hour >= self.london_hours[0]) & (hour < self.london_hours[1])]
        if int(hour.iloc[-1]) >= self.london_hours[1] and len(sess) >= 30:
            return {"LDNH": float(sess['high'].max()), "LDNL": float(sess['low'].min())}
        return {}

    def _round_number_levels(self, price, anchor=None):
        # [FIX 1.7] Grid is anchored to a stable reference (session/D1 open) so
        # levels don't vanish as price ticks into the next block.
        if anchor is None:
            anchor = price
        base = round(anchor / self.round_number_step) * self.round_number_step
        levels = {}
        for i in range(-5, 6):
            val = base + i * self.round_number_step
            fmt = f"{val:.4f}" if self.round_number_step < 1 else f"{int(val)}"
            levels[f"R_{fmt}"] = val
        return levels

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

        # [FIX 1.8] Precompute membership sets; single-pass filter.
        r_keys = {k for k in levels if k.startswith("R")}
        low_ok = set(self.LOW_LEVELS) | r_keys
        high_ok = set(self.HIGH_LEVELS) | r_keys

        low_hits = [(k, v) for k, v in levels.items()
                    if k in low_ok and bar['low'] < v < bar['close']]
        high_hits = [(k, v) for k, v in levels.items()
                     if k in high_ok and bar['close'] < v < bar['high']]

        if not low_hits and not high_hits:
            return None
        self.state.count("lvl_sweeps_detected")
        if low_hits and high_hits:
            self.state.count("lvl_ambiguous_bar")
            return None

        bullish = bool(low_hits)
        hits = low_hits if bullish else high_hits

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
        vpts = 2 if vr >= self.vol_mult_climax else (1 if vr >= self.vol_mult_climax - 0.5 else 0)
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
            if self.state.zone_touched(float(bar['low']), float(bar['high'])) is not None:
                score += 2; parts.append("H4zone+2")

        if any(a <= hour_now <= b for a, b in self.killzones):
            score += 1; parts.append("kz+1")

        grade = "A" if score >= self.a_score else ("B" if score >= self.min_score else None)
        risk = RISK_A_PCT if grade == "A" else (RISK_B_PCT if grade == "B" else 0.0)
        rr_used = 3.0 if grade == "A" else 2.0

        if not aligned:
            if grade:
                # [FIX 3.6] Renamed to make it clear these were hypothetical.
                self.state.count(f"lvl_would_be_counter_bias_{grade}")
                logger.info(f"[{self.name} SHADOW] Counter-bias {direction} {name} {level:.5f} "
                            f"(score {score}, grade {grade})")
            if not self.allow_counter_bias:
                self.state.count("lvl_rejected_counter_bias")
                return None

        if not bullish and not self.allow_shorts:
            return None

        breakdown = " ".join(parts)
        logger.info(f"[{self.name}] {direction} {name} {level:.5f}: score {score} [{breakdown}] -> {grade or 'no trade'}")
        if grade is None:
            self.state.count("lvl_below_min_score")
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
        m5 = self.fetch_data(bars=30)
        if m5 is None or len(m5) < 10:
            return
        last_t = m5['time'].iloc[-1]
        if last_t == self.last_m5_time:
            return
        self.last_m5_time = last_t

        bullish = p['direction'] == BULL_SWEEP
        bar = m5.iloc[-1]
        close = float(bar['close'])

        if (bullish and close < p['extreme']) or ((not bullish) and close > p['extreme']):
            self.state.count("lvl_invalidated_extreme")
            logger.info(f"[{self.name}] Setup invalidated: M5 closed beyond extreme {p['extreme']:.5f}")
            self._disarm()
            return

        if not self._choch(m5, p, bullish):
            self.state.update_agent_status(
                self.base_name, f"ARMED {p['direction']} {p['level_name']} ({p['score']}/{p['grade']}) — waiting CHoCH"
            )
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
            self.state.count("lvl_risk_too_wide")
            logger.info(f"[{self.name}] SL risk {risk:.5f} > {self.max_risk_atr} ATR; disarming.")
            self._disarm()
            return

        tag = f"Lvl:{p['level_name']}:{p['grade']}"
        logger.info(f"[{self.name}] M5 CHoCH confirmed for {p['level_name']} ({p['grade']}); executing.")
        self.state.count("lvl_trigger")

        if self.execute_trade(p['direction'], sl_raw, tag=tag,
                              risk_pct=p['risk'], rr_override=p['rr']):
            self.state.count(f"lvl_trades_{p['grade']}")
            self.state.count("lvl_trades")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                self._disarm()

    def _disarm(self):
        self.pending = None
        self.state.update_level_setup(None)
        if self.state.get_priority() == "level":
            self.state.set_priority(None)

    def step(self):
        if time.time() < self.cooldown_until:
            self.state.update_agent_status(self.base_name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            self.state.update_agent_status(self.base_name, f"Daily cap reached ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        bias, has_zone, fresh = self.state.get_macro()
        if bias not in ("BULLISH", "BEARISH") or not fresh:
            self.state.update_agent_status(self.base_name, "Idle (Macro stale)")
            return

        if self.cannot_open_new():   # [FIX 1.4]
            self.state.update_agent_status(self.base_name, "Idle (Position/cap)")
            return

        if self.pending is not None and time.time() > self.pending['expires_at']:
            self.state.count("lvl_setup_expired")
            self._disarm()

        self._refresh_htf()
        m15 = self.fetch_data(bars=130, timeframe=mt5.TIMEFRAME_M15)
        if m15 is None or len(m15) < 60:
            return

        levels = dict(self._htf)
        levels.update(self._london_levels(m15, broker_clock.offset))
        price = float(m15['close'].iloc[-1])
        # [FIX 1.7] Anchor the round-number grid to D1 open when available.
        anchor = self._htf.get("DO", price)
        levels.update(self._round_number_levels(price, anchor=anchor))
        self.state.update_levels(levels)

        bar_time = m15['time'].iloc[-1]
        if bar_time != self.last_m15_time:
            self.last_m15_time = bar_time
            setup = self._detect(m15, levels, bias)
            if setup is not None and (self.pending is None or setup['score'] > self.pending['score']):
                self.pending = setup
                self.state.set_priority("level")
                self.state.count("lvl_setups_armed")

        if self.pending is None:
            self.state.update_level_setup(None)
            self.state.update_agent_status(self.base_name, f"Scanning ({len(levels)} levels)")
            return

        p = self.pending
        self.state.update_level_setup({
            "direction": p['direction'], "level_name": p['level_name'], "level": p['level'],
            "score": p['score'], "grade": p['grade'], "breakdown": p['breakdown'],
            "expires_at": p['expires_at'],
        })
        self.state.update_agent_status(
            self.base_name, f"ARMED {p['direction']} {p['level_name']} ({p['score']}/{p['grade']})")
        self._check_trigger()


# ==========================================================================
# 5.6 FAILED BREAKOUT AGENT
# ==========================================================================
class FailedBreakoutAgent(ExecutionAgent):
    HTF_REFRESH_SEC = 600
    MIN_BREAK_ATR = 0.3

    def __init__(self, *args, allow_shorts=True, pending_ttl_sec=1800, killzones=((7, 10), (12, 15)),
                 max_risk_atr=2.5, max_trades_per_day=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_shorts = allow_shorts
        self.pending_ttl_sec = pending_ttl_sec
        self.killzones = killzones
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
        if not any(a <= hour_now <= b for a, b in self.killzones):
            return

        pdh = self._htf.get("PDH")
        pdl = self._htf.get("PDL")

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
        self.state.count("failbo_setups_armed")
        self.pending = {
            "direction": direction, "level_name": level_name,
            "level": float(level), "extreme": float(extreme),
            "atr": atr, "expires_at": time.time() + self.pending_ttl_sec,
            "attempts": 0,
        }
        self.state.set_priority("failbo")
        logger.info(f"[{self.name}] ARMED {direction} fade of broken {level_name} {level:.5f} "
                    f"(extreme {extreme:.5f})")

    def _disarm(self):
        self.pending = None
        self.state.update_failbo_setup(None)
        if self.state.get_priority() == "failbo":
            self.state.set_priority(None)

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

        if (bullish and close < p['extreme']) or ((not bullish) and close > p['extreme']):
            self.state.count("failbo_failed")
            logger.info(f"[{self.name}] Failed breakout invalidated.")
            self._disarm()
            return

        tol = 0.5 * p['atr']
        if bullish:
            touched = bar['low'] <= p['level'] + tol and close > p['level']
        else:
            touched = bar['high'] >= p['level'] - tol and close < p['level']
        if not touched:
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
            self.state.count("failbo_risk_too_wide")
            self._disarm()
            return

        logger.info(f"[{self.name}] M15 retest trigger for {p['level_name']} fade.")
        self.state.count("failbo_trigger")
        tag = f"FailBO:{p['level_name']}"
        if self.execute_trade(p['direction'], sl_raw, tag=tag,
                              risk_pct=RISK_B_PCT, rr_override=2.5):
            self.state.count("failbo_trades")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                self._disarm()

    def step(self):
        if time.time() < self.cooldown_until:
            self.state.update_agent_status(self.base_name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            self.state.update_agent_status(self.base_name, f"Daily cap ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        if self.pending is not None and time.time() > self.pending['expires_at']:
            self.state.count("failbo_expired")
            self._disarm()

        self._refresh_htf()

        if self.cannot_open_new() and self.pending is None:   # [FIX 1.4]
            self.state.update_agent_status(self.base_name, "Idle (Position/cap)")
            return

        self._scan_h1()

        if self.pending is None:
            self.state.update_failbo_setup(None)
            self.state.update_agent_status(self.base_name, "Scanning for failed breaks")
            return

        p = self.pending
        self.state.update_failbo_setup({
            "direction": p['direction'], "level_name": p['level_name'],
            "level": p['level'], "extreme": p['extreme'],
            "expires_at": p['expires_at'],
        })
        self.state.update_agent_status(
            self.base_name, f"ARMED fade {p['level_name']}")
        self._check_trigger()


# ==========================================================================
# 5.7 VWAP REVERSION AGENT
# ==========================================================================
class VWAPReversionAgent(ExecutionAgent):
    MIN_DIST_ATR = 2.5

    def __init__(self, *args, max_risk_atr=2.5, max_trades_per_day=2,
                 min_hour_utc=14, max_hour_utc=19, vol_mult=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_risk_atr = max_risk_atr
        self.max_trades_per_day = max_trades_per_day
        self.min_hour_utc = min_hour_utc
        self.max_hour_utc = max_hour_utc
        self.vol_mult = vol_mult
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
        if not (self.min_hour_utc <= hour_now <= self.max_hour_utc):
            return

        t_local_utc = m15['time'] - pd.Timedelta(hours=offset)
        date = t_local_utc.dt.date
        tp = (m15['high'] + m15['low'] + m15['close']) / 3.0
        vol = m15['tick_volume']
        cum_v = vol.groupby(date).cumsum()
        cum_tpv = (tp * vol).groupby(date).cumsum()
        vwap = float((cum_tpv / np.maximum(cum_v, 1)).iloc[-1])
        close = float(bar['close'])

        h1 = self.fetch_data(bars=20, timeframe=mt5.TIMEFRAME_H1)
        atr = compute_atr(h1, 14) if h1 is not None else 0.0
        if atr <= 0:
            return

        dist = close - vwap
        if abs(dist) < self.MIN_DIST_ATR * atr:
            return

        vr = volume_ratio(m15, len(m15) - 1)
        if vr < self.vol_mult:
            return

        bullish = dist < 0
        if bullish:
            reversal = bar['close'] > bar['open']
        else:
            reversal = bar['close'] < bar['open']
        if not reversal:
            return

        self.state.count("vwap_setups_armed")
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
                    f"(dist {dist:.5f}, vwap {vwap:.5f})")

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

        if (bullish and close < p['extreme']) or ((not bullish) and close > p['extreme']):
            self.state.count("vwap_failed")
            self._disarm()
            return

        # [FIX 3.5] Micro-CHoCH trigger — break of the *previous M5 bar* high/low.
        # Avoids chasing the top of the M15 reversion candle.
        if bullish:
            triggered = close > float(m5['high'].iloc[-2])
        else:
            triggered = close < float(m5['low'].iloc[-2])

        if not triggered:
            self.state.update_agent_status(
                self.base_name,
                f"ARMED VWAP {p['direction']} — need close "
                f"{'>' if bullish else '<'} "
                f"{(float(m5['high'].iloc[-2]) if bullish else float(m5['low'].iloc[-2])):.5f} "
                f"(now {close:.5f})"
            )
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
            self.state.count("vwap_risk_too_wide")
            self._disarm()
            return

        vwap_target_dist = abs(p['vwap'] - entry)
        rr_needed = vwap_target_dist / risk if risk > 0 else 0
        rr_used = max(1.5, min(3.0, rr_needed))

        logger.info(f"[{self.name}] M5 trigger for VWAP reversion (rr~{rr_used:.2f}).")
        self.state.count("vwap_trigger")
        if self.execute_trade(p['direction'], sl_raw, tag="VWAPrev",
                              risk_pct=RISK_B_PCT, rr_override=rr_used):
            self.state.count("vwap_trades")
            self.trades_today += 1
            self.cooldown_until = time.time() + self.cooldown_sec
            self._disarm()
        else:
            p['attempts'] += 1
            if p['attempts'] >= 2:
                self._disarm()

    def _disarm(self):
        self.pending = None
        self.state.update_vwap_setup(None)

    def step(self):
        if time.time() < self.cooldown_until:
            self.state.update_agent_status(self.base_name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        today = time.strftime('%Y-%m-%d', time.gmtime())
        if self.trades_date != today:
            self.trades_date, self.trades_today = today, 0
        if self.trades_today >= self.max_trades_per_day:
            self.state.update_agent_status(self.base_name, f"Daily cap ({self.max_trades_per_day})")
            return

        broker_clock.refresh(self.symbol)
        if self.pending is not None and time.time() > self.pending['expires_at']:
            self.state.count("vwap_expired")
            self._disarm()

        if self.cannot_open_new() and self.pending is None:   # [FIX 1.4]
            self.state.update_agent_status(self.base_name, "Idle (Position/cap)")
            return

        if self.pending is None:
            self._scan_m15()

        if self.pending is None:
            self.state.update_vwap_setup(None)
            self.state.update_agent_status(self.base_name, "Scanning VWAP distance")
            return

        p = self.pending
        self.state.update_vwap_setup({
            "direction": p['direction'], "extreme": p['extreme'],
            "vwap": p['vwap'], "dist": p['dist'], "expires_at": p['expires_at'],
        })
        self.state.update_agent_status(self.base_name, f"ARMED VWAP {p['direction']}")
        self._check_trigger()


# ==========================================================================
# 6. TRADE MANAGER — 3-STAGE EXIT + TRAILING + EOD FLAT
# ==========================================================================
class TradeManagerAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, state, system_magics,
                 be_trigger_r=1.0, be_buffer_points=80,
                 partial1_r=1.0, partial2_r=2.0,
                 trail_lookback_m15=3, flat_at_utc=FLAT_AT_UTC):
        super().__init__(name, symbol, timeframe, sleep_interval, state)
        self.system_magics = system_magics
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
        # [FIX 1.3] Retry-based EOD flat instead of once-per-day latch.
        self._last_flat_attempt_ts = 0.0
        self._flat_attempt_interval = 30.0

    def _fetch_m15_swing(self, is_buy):
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 6)
        if rates is None or len(rates) < 4:
            return None
        df = pd.DataFrame(rates).iloc[:-1]
        if is_buy:
            return float(df['low'].iloc[-self.trail_lookback_m15:].min())
        return float(df['high'].iloc[-self.trail_lookback_m15:].max())

    @staticmethod
    def _round_volume(vol, info):
        step = info.volume_step
        v = math.floor(vol / step + 1e-9) * step
        return round(round(v / step) * step, 8)

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

    def _safe_partial_volume(self, pos, fraction, info):
        # [FIX 1.10] Compute a partial that leaves >= volume_min behind.
        step = info.volume_step
        vmin = info.volume_min
        desired = self._round_volume(pos.volume * fraction, info)
        if desired < vmin:
            return None
        remainder = pos.volume - desired
        if remainder < vmin:
            desired = self._round_volume(pos.volume - vmin, info)
            if desired < vmin:
                return None
        if desired >= pos.volume:
            return None
        return desired

    def _force_flat_check(self, positions, info):
        # [FIX 1.3] Use broker time; retry until flat; back off to 30s.
        n = now_broker()
        if not (n.hour > self.flat_at_utc or
                (n.hour == self.flat_at_utc and n.minute >= FLAT_AT_MINUTE)):
            return
        live = [p for p in positions if p.magic in self.system_magics]
        if not live:
            return
        now_ts = time.time()
        if now_ts - self._last_flat_attempt_ts < self._flat_attempt_interval:
            return
        self._last_flat_attempt_ts = now_ts
        for pos in live:
            logger.warning(f"[{self.name}] EOD flat: closing ticket {pos.ticket} (magic {pos.magic}).")
            self._partial_close(pos, pos.volume, info, "EOD_flat")
            log_event("eod_flat", symbol=self.symbol, ticket=pos.ticket, magic=pos.magic)
            self.state.count("eod_flat_closes")

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
            if new_sl <= pos.price_open:
                new_sl = pos.price_open + buffer
            if new_sl <= pos.sl:
                return
        else:
            new_sl = swing + buffer
            if new_sl >= pos.price_open:
                new_sl = pos.price_open - buffer
            if new_sl >= pos.sl:
                return

        if (is_buy and new_sl >= pos.price_current) or ((not is_buy) and new_sl <= pos.price_current):
            return
        if abs(pos.price_current - new_sl) < info.trade_stops_level * info.point:
            return
        if self._modify_sl(pos, new_sl, info):
            logger.info(f"[{self.name}] Trailed ticket {pos.ticket} to {new_sl:.5f}")

    def step(self):
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            info = mt5.symbol_info(self.symbol)
        if positions is None or info is None:
            return

        mine = [p for p in positions if p.magic in self.system_magics]
        live = {p.ticket for p in mine}
        self.state.update_agent_status(self.base_name, f"Managing {len(mine)} position(s)")

        self._force_flat_check(mine, info)

        point, digits = info.point, info.digits
        stops_dist = info.trade_stops_level * point
        buffer = self.be_buffer_points * point

        for pos in mine:
            if pos.sl == 0.0 or pos.tp == 0.0:
                continue
            is_buy = pos.type == mt5.POSITION_TYPE_BUY

            if pos.ticket not in self._risks:
                on_loss_side = (pos.sl < pos.price_open) if is_buy else (pos.sl > pos.price_open)
                if on_loss_side:
                    self._risks[pos.ticket] = abs(pos.price_open - pos.sl)
                else:
                    self._risks[pos.ticket] = max(abs(pos.tp - pos.price_open) / 2.0, point * 100)
            if pos.ticket not in self._originals:
                self._originals[pos.ticket] = pos.volume

            risk = self._risks[pos.ticket]
            original = self._originals[pos.ticket]
            if risk <= 0 or original <= 0:
                continue
            profit = (pos.price_current - pos.price_open) if is_buy \
                     else (pos.price_open - pos.price_current)

            if pos.ticket not in self._partial1_done and profit >= risk * self.partial1_r:
                vol = self._safe_partial_volume(pos, 0.40, info)   # [FIX 1.10]
                if vol is not None:
                    if self._partial_close(pos, vol, info, "P1@1R"):
                        self._partial1_done.add(pos.ticket)
                        logger.info(f"[{self.name}] Ticket {pos.ticket} P1 40% @ 1R ({vol}).")
                        self.state.count("partials_1R")

            if pos.ticket not in self._partial2_done and profit >= risk * self.partial2_r:
                vol = self._safe_partial_volume(pos, 0.30, info)   # [FIX 1.10]
                if vol is not None:
                    if self._partial_close(pos, vol, info, "P2@2R"):
                        self._partial2_done.add(pos.ticket)
                        logger.info(f"[{self.name}] Ticket {pos.ticket} P2 30% @ 2R ({vol}).")
                        self.state.count("partials_2R")

            secured = (pos.sl >= pos.price_open) if is_buy else (pos.sl <= pos.price_open)
            if not secured and profit >= risk * self.be_trigger_r:
                new_sl = round(pos.price_open + buffer, digits) if is_buy \
                         else round(pos.price_open - buffer, digits)
                if not ((is_buy and new_sl >= pos.price_current) or
                        ((not is_buy) and new_sl <= pos.price_current)):
                    if abs(pos.price_current - new_sl) >= stops_dist:
                        if self._modify_sl(pos, new_sl, info):
                            logger.info(f"[{self.name}] Ticket {pos.ticket} -> BE ({new_sl}).")
                            self.state.count("be_moves")

            self._trail_runner(pos, info, risk)

        self._risks = {t: v for t, v in self._risks.items() if t in live}
        self._originals = {t: v for t, v in self._originals.items() if t in live}
        self._partial1_done = {t for t in self._partial1_done if t in live}
        self._partial2_done = {t for t in self._partial2_done if t in live}


# ==========================================================================
# 7. TELEMETRY AGENT
# ==========================================================================
class TelemetryAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, state, system_magics):
        super().__init__(name, symbol, timeframe, sleep_interval, state)
        self.system_magics = system_magics

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
            if p.magic in self.system_magics:
                trades.append({
                    "ticket": p.ticket, "agent": p.comment,
                    "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": p.volume, "open": p.price_open, "current": p.price_current,
                    "sl": p.sl, "tp": p.tp, "profit": p.profit,
                })

        self.state.update_snapshot(market, account, trades)
        self.state.update_agent_status(self.base_name, "Snapshotting MT5 state")


# ==========================================================================
# 7.5 CLOSED TRADES (dashboard)  [minor]
# ==========================================================================
def recent_closed_trades(symbol, magics, limit=50):
    now_ts = int(time.time())
    with mt5_lock:
        deals = mt5.history_deals_get(now_ts - 7 * 86400, now_ts + 1)
    if deals is None:
        return []
    out = []
    for d in reversed(deals):
        if d.symbol != symbol:
            continue
        if d.magic not in magics:
            continue
        # DEAL_ENTRY_OUT = 1
        if getattr(d, "entry", 0) not in (1,):
            continue
        out.append({
            "ticket": d.ticket,
            "time": d.time,
            "magic": d.magic,
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "swap": d.swap,
            "commission": d.commission,
            "comment": d.comment,
        })
        if len(out) >= limit:
            break
    return out


# ==========================================================================
# 8. WEB DASHBOARD
# ==========================================================================
app = FastAPI(title="Wyckoff Bot Terminal")


@app.get("/api/state")
def get_system_state():
    return {sym: state.export() for sym, state in symbol_states.items()}


@app.get("/api/closed")
def get_closed():
    # [minor] Closed trades panel data source.
    out = {}
    for sym, state in symbol_states.items():
        magics = {m for m in ALL_SYSTEM_MAGICS}
        out[sym] = recent_closed_trades(sym, magics, limit=30)
    return out


@app.post("/api/halt")
def halt_trading():
    # [FIX 4.8] Kill switch — stops all new entries immediately.
    trading_enabled.set(False)
    logger.warning("!! GLOBAL HALT engaged via API. !!")
    log_event("halt")
    return {"enabled": False}


@app.post("/api/resume")
def resume_trading():
    trading_enabled.set(True)
    logger.warning("!! GLOBAL HALT released via API. !!")
    log_event("resume")
    return {"enabled": True}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Symbol Command Terminal v9</title>
<style>
:root{--bg:#0c0d0e;--card:#16181b;--line:#262a2f;--fg:#e1e3e6;--mut:#8b9098;--bull:#3fb950;--bear:#f85149;--warn:#d29922;--info:#58a6ff}
*{box-sizing:border-box}
body{margin:0;padding:20px;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
select{padding:4px;border-radius:4px;background:var(--card);color:var(--fg);border:1px solid var(--line);font-size:14px;outline:none}
button{padding:5px 10px;border-radius:4px;background:#30363d;color:var(--fg);border:1px solid var(--line);cursor:pointer;font-size:13px}
button:hover{background:#3c444d}
button.danger{background:#7d2029;color:#fff;border-color:#a8323d}
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
<h1>Multi-Symbol Command Terminal v9</h1>
<div class="sub">
  Symbol: <select id="sym_sel" onchange="renderCurrent()"></select>
  <button id="halt_btn" onclick="toggleHalt()">HALT</button>
  <span id="conn">connecting...</span>
</div>
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
  <div class="card wide"><h2>Recent closed trades</h2><div id="closed"></div></div>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=5) => (v === null || v === undefined || isNaN(v)) ? '--' : Number(v).toFixed(d);
const row = (k, v, cls='') => `<div class="row"><span>${esc(k)}</span><span class="${cls}">${v}</span></div>`;
const pnl = v => (v === null || v === undefined) ? '--' : `<span class="${v >= 0 ? 'bull' : 'bear'}">${num(v, 2)}</span>`;

let globalData = {};
let globalClosed = {};
let halted = false;

function updateSelect(data) {
  const sel = $('sym_sel');
  const symbols = Object.keys(data).sort();
  const current = sel.value;
  if (sel.options.length !== symbols.length) {
    sel.innerHTML = symbols.map(s => `<option value="${s}">${s}</option>`).join('');
    if (symbols.includes(current)) sel.value = current;
    else if (symbols.length) sel.value = symbols[0];
  }
}

async function toggleHalt() {
  try {
    const r = await fetch(halted ? '/api/resume' : '/api/halt', {method: 'POST'});
    const j = await r.json();
    halted = !j.enabled;
    updateHaltBtn();
  } catch(e) {}
}
function updateHaltBtn() {
  const b = $('halt_btn');
  if (halted) { b.textContent = 'RESUME'; b.classList.add('danger'); }
  else        { b.textContent = 'HALT';   b.classList.remove('danger'); }
}

function renderCurrent() {
  if (!globalData || !Object.keys(globalData).length) return;
  const sym = $('sym_sel').value;
  if (!sym || !globalData[sym]) return;
  const d = globalData[sym];
  if (typeof d.trading_enabled === 'boolean') { halted = !d.trading_enabled; updateHaltBtn(); }

  const m = d.market || {}, a = d.account || {}, mc = d.macro || {}, s = d.session || {};

  $('market').innerHTML =
    row('Symbol', esc(m.symbol || '--')) +
    row('Bid / Ask', `${num(m.bid)} / ${num(m.ask)}`) +
    row('Spread', `${num(m.spread,1)} pts`) +
    row('Trading hours', d.trading_hours_open ? '<span class="badge on">OPEN</span>' : '<span class="badge off">CLOSED</span>') +
    row('News blackout', d.news_blackout ? '<span class="badge hot">YES</span>' : 'no');

  $('account').innerHTML =
    row('Balance', '$' + num(a.balance, 2)) +
    row('Equity', '$' + num(a.equity, 2)) +
    row('Free margin', '$' + num(a.margin_free, 2)) +
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
    row('Vol ratio', num(s.vol_ratio, 2) + 'x') +
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
        `<td>${num(x.volume, 2)}</td><td>${num(x.open)}</td><td>${num(x.current)}</td>` +
        `<td>${num(x.sl)}</td><td>${num(x.tp)}</td><td>${pnl(x.profit)}</td></tr>`).join('') + '</table>'
    : '<div class="empty">No open positions.</div>';

  const ct = (globalClosed[sym] || []);
  $('closed').innerHTML = ct.length
    ? '<table><tr><th>Time</th><th>Ticket</th><th>Magic</th><th>Lots</th><th>Price</th><th>Profit</th><th>Comment</th></tr>' +
      ct.map(x => {
        const dt = new Date(x.time * 1000).toLocaleString();
        return `<tr><td>${esc(dt)}</td><td>${esc(x.ticket)}</td><td>${esc(x.magic)}</td>` +
               `<td>${num(x.volume,2)}</td><td>${num(x.price)}</td>` +
               `<td>${pnl((x.profit||0)+(x.swap||0)+(x.commission||0))}</td>` +
               `<td>${esc(x.comment)}</td></tr>`;
      }).join('') + '</table>'
    : '<div class="empty">No closed trades yet.</div>';

  const age = d.snapshot_age;
  $('conn').innerHTML = 'live · ' + new Date().toLocaleTimeString() +
    (age !== null && age > 15 ? ' · <span class="bear">snapshot '+age+'s old</span>' : '');
}

async function tick() {
  try {
    const [r1, r2] = await Promise.all([
      fetch('/api/state', {cache: 'no-store'}),
      fetch('/api/closed', {cache: 'no-store'}),
    ]);
    if (!r1.ok) throw new Error(r1.status);
    globalData = await r1.json();
    if (r2.ok) globalClosed = await r2.json();
    updateSelect(globalData);
    renderCurrent();
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
# 9. ORCHESTRATOR & SYMBOL CONFIGURATION
# ==========================================================================
def get_symbol_config(sym):
    cfg = {
        "round_number_step": 0.0050,
        "max_spread_points": 20.0,
        "vol_mult_base": 1.5,
        "vol_mult_climax": 2.0,
        "asian_end_hour": 6,
        "london_hours": (7, 12),
        "killzones": ((7, 10), (13, 16)),
    }
    if "JPY" in sym:
        cfg["round_number_step"] = 0.50
        cfg["max_spread_points"] = 25.0
        cfg["asian_end_hour"] = 4
    elif "SEK" in sym or "CNH" in sym:
        cfg["round_number_step"] = 0.05
        cfg["max_spread_points"] = 150.0
    elif "XAU" in sym:
        cfg["round_number_step"] = 25.0
        cfg["max_spread_points"] = 50.0
        cfg["vol_mult_climax"] = 2.5
        cfg["killzones"] = ((7, 10), (12, 15))
    return cfg


def resolve_symbol_suffix(symbols):
    # [FIX 4.2] Detect a uniform broker suffix from whichever symbol carries one.
    suffix = ""
    for s in symbols:
        if "." in s:
            suffix = "." + s.split(".", 1)[1]
            break
    if not suffix:
        return symbols
    with mt5_lock:
        all_names = {s.name for s in (mt5.symbols_get() or [])}
    out = []
    for s in symbols:
        if "." in s or (s + suffix) not in all_names:
            out.append(s)
        else:
            out.append(s + suffix)
            logger.info(f"Applied suffix '{suffix}' to symbol {s} -> {s + suffix}")
    return out


if __name__ == "__main__":
    missing = [n for n in REQUIRED_MT5_NAMES if not hasattr(mt5, n)]
    if missing:
        logger.error(f"MT5 package missing: {missing}")
        quit()

    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        quit()

    # [FIX 4.2] Uniform symbol names
    #TARGET_SYMBOLS = resolve_symbol_suffix(TARGET_SYMBOLS)

    # [FIX 4.1] Load magic bases and assign
    _load_magic_bases()
    all_agents = []

    for i, sym in enumerate(TARGET_SYMBOLS):
        if not mt5.symbol_select(sym, True):
            logger.error(f"Cannot select {sym}: {mt5.last_error()}")
            continue

        state = TradingState()
        symbol_states[sym] = state

        cfg = get_symbol_config(sym)

        # [FIX 4.1] Stable magic bases across restarts
        prefix = get_or_create_symbol_base(sym)
        magics = {k: prefix + slot for k, slot in MAGIC_SLOT.items()}
        sys_magics = set(magics.values())
        ALL_SYSTEM_MAGICS.update(sys_magics)   # [FIX 2.1]

        macro = MacroAgent("Macro_H4", sym, mt5.TIMEFRAME_H4, 60, state)

        killzone = KillzoneAgent(
            "Killzone_M15", sym, mt5.TIMEFRAME_M15, 15, state, clock=broker_clock,
            asian_end_hour=cfg["asian_end_hour"], killzones=cfg["killzones"],
            vol_mult=cfg["vol_mult_base"], allow_shorts=True, sweep_ttl_sec=3600
        )

        kz_sweep = KillzoneSweepAgent(
            "KZSweep", sym, mt5.TIMEFRAME_M15, 15, state,
            magic=magics["KZ_SWEEP"], system_magics=sys_magics,
            use_dynamic_lot=True, fixed_lot=0.1, risk_pct=RISK_B_PCT,
            rr=2.5, deviation=20, cooldown_sec=1800, min_sl_spread_mult=3.0,
            max_spread_points=cfg["max_spread_points"], max_risk_atr=2.0, retry_throttle_sec=90
        )

        swing = SwingAgent(
            "Swing_H1", sym, mt5.TIMEFRAME_H1, 15, state,
            magic=magics["SWING"], system_magics=sys_magics,
            use_dynamic_lot=True, fixed_lot=0.1, risk_pct=RISK_B_PCT,
            rr=2.0, deviation=20, cooldown_sec=3600, min_sl_spread_mult=3.0,
            max_spread_points=cfg["max_spread_points"], allow_shorts=True,
            sweep_lookback=6, swing_n=30, vol_mult=cfg["vol_mult_base"]
        )

        levelsweep = LevelSweepAgent(
            "LevelSweep", sym, mt5.TIMEFRAME_M5, 10, state,
            magic=magics["LEVEL"], system_magics=sys_magics,
            use_dynamic_lot=True, fixed_lot=0.1, risk_pct=RISK_A_PCT,
            rr=2.5, deviation=20, cooldown_sec=900, min_sl_spread_mult=3.0,
            max_spread_points=cfg["max_spread_points"], allow_shorts=True, allow_counter_bias=False,
            min_score=5, a_score=7, pending_ttl_sec=2700, choch_lookback=6,
            max_risk_atr=3.0, max_trades_per_day=4,
            london_hours=cfg["london_hours"], killzones=cfg["killzones"],
            round_number_step=cfg["round_number_step"], vol_mult_climax=cfg["vol_mult_climax"]
        )

        failedbo = FailedBreakoutAgent(
            "FailedBO", sym, mt5.TIMEFRAME_M15, 20, state,
            magic=magics["FAILBO"], system_magics=sys_magics,
            use_dynamic_lot=True, fixed_lot=0.05, risk_pct=RISK_B_PCT,
            rr=2.5, deviation=20, cooldown_sec=1200, min_sl_spread_mult=3.0,
            max_spread_points=cfg["max_spread_points"], allow_shorts=True, pending_ttl_sec=1800,
            killzones=cfg["killzones"], max_risk_atr=2.5, max_trades_per_day=2
        )

        vwaprev = VWAPReversionAgent(
            "VWAPRev", sym, mt5.TIMEFRAME_M5, 15, state,
            magic=magics["VWAP"], system_magics=sys_magics,
            use_dynamic_lot=True, fixed_lot=0.05, risk_pct=RISK_B_PCT,
            rr=2.0, deviation=20, cooldown_sec=1200, min_sl_spread_mult=3.0,
            max_spread_points=cfg["max_spread_points"], max_risk_atr=2.5, max_trades_per_day=2,
            min_hour_utc=14, max_hour_utc=19, vol_mult=cfg["vol_mult_climax"]
        )

        continuation = ContinuationAgent(
            "Continuation_M15", sym, mt5.TIMEFRAME_M5, 15, state,
            magic=magics["PYRAMID"], system_magics=sys_magics,
            use_dynamic_lot=True, fixed_lot=0.05, risk_pct=RISK_PYRAMID,
            rr=2.5, deviation=20, cooldown_sec=900, min_sl_spread_mult=3.0,
            max_spread_points=cfg["max_spread_points"], parent_magic=magics["LEVEL"]
        )

        manager = TradeManagerAgent(
            "Manager", sym, mt5.TIMEFRAME_M1, 3, state, system_magics=sys_magics,
            be_trigger_r=1.0, be_buffer_points=80,
            partial1_r=1.0, partial2_r=2.0, trail_lookback_m15=3
        )

        telemetry = TelemetryAgent("Telemetry", sym, mt5.TIMEFRAME_M1, 2, state, sys_magics)

        all_agents.extend([
            macro, killzone, kz_sweep, swing, levelsweep,
            failedbo, vwaprev, continuation, manager, telemetry
        ])

    # Prime the broker clock before threads start.
    if symbol_states:
        broker_clock.refresh(next(iter(symbol_states.keys())))

    for a in all_agents:
        a.start()

    threading.Thread(target=run_web_server, name="WebServer", daemon=True).start()

    warned = set()
    terminal_was_ok = True

    try:
        logger.info(f"System online tracking {len(symbol_states)} symbols. "
                    f"Dashboard: http://127.0.0.1:8000 (Ctrl+C to stop)")
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

            broker_clock.warn_if_unverified()

            for a in all_agents:
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
        for a in all_agents:
            a.stop()
        for a in all_agents:
            a.join(timeout=10)
        mt5.shutdown()
        logger.info("Shutdown complete.")