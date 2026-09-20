import math
import threading
import time
import traceback
from dataclasses import dataclass

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ==========================================
# 0. CONSTANTS & GLOBAL MT5 LOCK
# ==========================================
MAGIC = 999999
BULL_SWEEP = "BULLISH_SWEEP"
BEAR_SWEEP = "BEARISH_SWEEP"

# The MetaTrader5 python package is not documented as thread-safe, so EVERY
# mt5.* call in every thread goes through this one re-entrant lock.
mt5_lock = threading.RLock()

import logging
from logging.handlers import RotatingFileHandler
import os

# ==========================================
# 0.5 LOGGER CONFIGURATION
# ==========================================
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Creates a rotating log: max 5MB per file, keeps the last 3 backups.
log_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "wyckoff_bot.log"), 
    maxBytes=5*1024*1024, 
    backupCount=3
)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)

# Set up the root logger
logger = logging.getLogger("WyckoffBot")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Optional: Add a StreamHandler if you still want a clean output in the CLI
cli_handler = logging.StreamHandler()
cli_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(cli_handler)

# ==========================================
# 1. SHARED STATE
# ==========================================
@dataclass(frozen=True)
class Sweep:
    direction: str          # BULL_SWEEP / BEAR_SWEEP
    level: float            # swing level that was swept
    extreme: float          # spring low (bull) / upthrust high (bear)
    bar_time: pd.Timestamp  # open time of the sweep bar
    close_time: pd.Timestamp  # close time of the sweep bar
    expires_at: float       # unix time after which the sweep is stale


class TradingState:
    def __init__(self):
        self.macro_bias = "NEUTRAL"
        self.macro_poi_active = False
        self.macro_ts = 0.0
        self._sweep = None
        self.state_lock = threading.Lock()

    # --- macro ---
    def update_macro(self, bias, in_poi):
        with self.state_lock:
            self.macro_bias = bias
            self.macro_poi_active = in_poi
            self.macro_ts = time.time()

    def get_macro(self, max_age=600):
        """Returns (bias, in_poi, fresh). Stale macro data is reported as not fresh."""
        with self.state_lock:
            fresh = (time.time() - self.macro_ts) <= max_age
            return self.macro_bias, self.macro_poi_active, fresh

    # --- sweep (latched until consumed / invalidated / expired) ---
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


shared_state = TradingState()


# ==========================================
# 2. BASE AGENT
# ==========================================
class BaseAgent(threading.Thread):
    banner = ""

    def __init__(self, name, symbol, timeframe, sleep_interval):
        super().__init__(name=name, daemon=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.sleep_interval = sleep_interval
        self._stop_event = threading.Event()
        self.last_beat = time.time()

    @property
    def running(self):
        return not self._stop_event.is_set()

    def fetch_data(self, bars=100, closed_only=True):
        """Returns a DataFrame with a clean 0..n-1 index.
        closed_only=True drops the still-forming candle."""
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
        logger.info(f"[{self.name}] {self.banner}")
        while not self._stop_event.is_set():
            self.last_beat = time.time()
            try:
                self.step()
            except Exception:
                logger.info(f"[{self.name}] Unhandled error in step():")
                traceback.print_exc()
            self._stop_event.wait(self.sleep_interval)

    def stop(self):
        self._stop_event.set()


# ==========================================
# 3. MACRO AGENT (H4) - BIAS & POI
# ==========================================
class MacroAgent(BaseAgent):
    banner = "Tracking Institutional Bias..."

    EXPANSION_MULT = 1.5
    ATR_PERIOD = 14
    MAX_ZONE_AGE = 100  # bars

    def step(self):
        df = self.fetch_data(bars=300)  # closed bars only
        if df is None or len(df) < 210:
            return

        # 1. Macro trend (EMA 50 / 200)
        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        ema200 = df['close'].ewm(span=200, adjust=False).mean()
        bias = "BULLISH" if ema50.iloc[-1] > ema200.iloc[-1] else "BEARISH"

        # 2. True range + ATR (previous bars only, so the expansion candle
        #    is not part of its own baseline)
        prev_close = df['close'].shift(1)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum((df['high'] - prev_close).abs(),
                       (df['low'] - prev_close).abs()),
        )
        atr = tr.rolling(self.ATR_PERIOD).mean().shift(1)
        big = tr > self.EXPANSION_MULT * atr

        shared_state.update_macro(bias, self._in_poi(df, bias, big))

    def _in_poi(self, df, bias, big):
        """BULLISH -> demand zone (candle before last bullish expansion).
        BEARISH -> supply zone (candle before last bearish expansion)."""
        bullish = bias == "BULLISH"
        direction = (df['close'] > df['open']) if bullish else (df['close'] < df['open'])
        idx = np.flatnonzero((direction & big).to_numpy())
        idx = idx[idx >= 1]
        if len(idx) == 0:
            return False

        i = int(idx[-1])
        n = len(df)
        if (n - 1) - i > self.MAX_ZONE_AGE:
            return False

        zone_high = df['high'].iloc[i - 1]
        zone_low = df['low'].iloc[i - 1]

        # Invalidation: any close beyond the zone after it formed
        after = df['close'].iloc[i + 1:]
        if bullish and (after < zone_low).any():
            return False
        if not bullish and (after > zone_high).any():
            return False

        price = df['close'].iloc[-1]
        return bool(zone_low <= price <= zone_high)


# ==========================================
# 4. INTERMEDIATE AGENT (M15) - WYCKOFF SPRING / UPTHRUST
# ==========================================
class IntermediateAgent(BaseAgent):
    banner = "Hunting for Liquidity Sweeps..."

    def __init__(self, name, symbol, timeframe, sleep_interval,
                 allow_shorts=False, sweep_ttl_sec=3600):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.allow_shorts = allow_shorts
        self.sweep_ttl_sec = sweep_ttl_sec
        self.last_bar_time = None

    def step(self):
        bias, in_poi, fresh = shared_state.get_macro()
        if not (fresh and in_poi):
            return
        if bias not in ("BULLISH", "BEARISH"):
            return
        if bias == "BEARISH" and not self.allow_shorts:
            return
        if shared_state.get_sweep() is not None:
            return  # a sweep is already latched; wait for Micro to use/expire it

        df = self.fetch_data(bars=60)  # closed bars only
        if df is None or len(df) < 50:
            return

        bar = df.iloc[-1]  # last CLOSED M15 bar is the candidate sweep bar
        if bar['time'] == self.last_bar_time:
            return  # already evaluated this bar
        self.last_bar_time = bar['time']

        vol_ma = df['tick_volume'].iloc[-21:-1].mean()
        climax = bar['tick_volume'] > (vol_ma * 1.5)

        if bias == "BULLISH":
            # Spring: spikes below swing low, closes back above, bullish candle, climax volume
            level = df['low'].iloc[-45:-5].min()
            hit = (bar['low'] < level and bar['close'] > level
                   and bar['close'] > bar['open'] and climax)
            direction, extreme = BULL_SWEEP, float(bar['low'])
        else:
            # Upthrust: spikes above swing high, closes back below, bearish candle, climax volume
            level = df['high'].iloc[-45:-5].max()
            hit = (bar['high'] > level and bar['close'] < level
                   and bar['close'] < bar['open'] and climax)
            direction, extreme = BEAR_SWEEP, float(bar['high'])

        if hit:
            bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
            shared_state.set_sweep(Sweep(
                direction=direction,
                level=float(level),
                extreme=extreme,
                bar_time=bar['time'],
                close_time=bar['time'] + bar_delta,
                expires_at=time.time() + self.sweep_ttl_sec,
            ))
            logger.info(f"[{self.name}] {direction} latched at {bar['time']} "
                  f"(level {level:.2f}, extreme {extreme:.2f})")


# ==========================================
# 5. MICRO AGENT (M1) - FVG & EXECUTION
# ==========================================
class MicroAgent(BaseAgent):
    banner = "Waiting for Liquidity Sweeps to execute FVGs..."

    def __init__(self, name, symbol, timeframe, sleep_interval,
                 use_dynamic_lot, fixed_lot, risk_pct,
                 rr=2.0, deviation=20, cooldown_sec=300,
                 min_sl_spread_mult=3.0, max_spread_points=None):
        super().__init__(name, symbol, timeframe, sleep_interval)
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

    # ---------- helpers (call with mt5_lock held or via execute_trade) ----------
    def has_open_position(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return True  # fail closed: if we can't tell, assume one is open
        return any(p.magic == MAGIC for p in positions)

    @staticmethod
    def _filling_mode(info):
        if info.filling_mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def calculate_lot_size(self, info, entry, sl):
        """Returns a valid lot, or None if the trade should be skipped."""
        step = info.volume_step
        vmin, vmax = info.volume_min, info.volume_max

        if not self.use_dynamic_lot:
            lot = math.floor(self.fixed_lot / step + 1e-9) * step
            lot = max(vmin, min(lot, vmax))
            return round(round(lot / step) * step, 8)

        acct = mt5.account_info()
        if acct is None:
            logger.info(f"[{self.name}] No account info; skipping trade.")
            return None

        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value_loss or info.trade_tick_value
        if tick_size <= 0 or tick_value <= 0:
            logger.info(f"[{self.name}] Bad tick size/value; skipping trade.")
            return None

        risk_amount = acct.balance * (self.risk_pct / 100.0)
        risk_per_lot = (abs(entry - sl) / tick_size) * tick_value
        if risk_per_lot <= 0:
            return None

        raw_lot = risk_amount / risk_per_lot
        lot = math.floor(raw_lot / step + 1e-9) * step

        if lot < vmin:
            logger.info(f"[{self.name}] Risk-based lot {raw_lot:.4f} < min {vmin}; "
                  f"skipping (would exceed {self.risk_pct}% risk).")
            return None

        lot = min(lot, vmax)
        return round(round(lot / step) * step, 8)

    def execute_trade(self, direction, sl_raw):
        """Returns True only if an order was actually filled."""
        is_buy = direction == BULL_SWEEP

        with mt5_lock:
            if self.has_open_position():
                return False

            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)
            if info is None or tick is None:
                logger.info(f"[{self.name}] No symbol/tick info: {mt5.last_error()}")
                return False

            spread = tick.ask - tick.bid
            if (self.max_spread_points is not None
                    and spread / info.point > self.max_spread_points):
                logger.info(f"[{self.name}] Spread too wide ({spread / info.point:.0f} pts); skipping.")
                return False

            entry = tick.ask if is_buy else tick.bid
            sl = sl_raw - spread if is_buy else sl_raw + spread

            if (is_buy and sl >= entry) or (not is_buy and sl <= entry):
                logger.info(f"[{self.name}] SL on wrong side of entry; skipping.")
                return False

            risk = abs(entry - sl)
            min_dist = max(info.trade_stops_level * info.point,
                           spread * self.min_sl_spread_mult)
            if risk < min_dist:
                logger.info(f"[{self.name}] SL distance {risk:.2f} < min {min_dist:.2f}; skipping.")
                return False

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
                "magic": MAGIC,
                "comment": "Wyckoff_FVG_Entry",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(info),
            }
            result = mt5.order_send(request)

        if result is None:
            logger.info(f"[{self.name}] order_send returned None: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.info(f"[{self.name}] Trade failed: retcode={result.retcode} {result.comment}")
            return False

        logger.info(f"[{self.name}] Executed {'BUY' if is_buy else 'SELL'} {lot} lots at "
              f"{result.price}. SL: {request['sl']}, TP: {request['tp']}")
        return True

    # ---------- main loop body ----------
    def step(self):
        if time.time() < self.cooldown_until:
            return

        sweep = shared_state.get_sweep()  # None if absent or expired
        if sweep is None:
            return

        with mt5_lock:
            if self.has_open_position():
                return

        df = self.fetch_data(bars=10)  # closed bars only
        if df is None or len(df) < 5:
            return

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c3['time'] == self.last_bar_time:
            return  # evaluate each closed M1 bar once
        self.last_bar_time = c3['time']

        bullish = sweep.direction == BULL_SWEEP

        # Invalidate a failed spring/upthrust
        if bullish and c3['close'] < sweep.extreme:
            logger.info(f"[{self.name}] Spring failed (close below sweep low); sweep cleared.")
            shared_state.clear_sweep()
            return
        if not bullish and c3['close'] > sweep.extreme:
            logger.info(f"[{self.name}] Upthrust failed (close above sweep high); sweep cleared.")
            shared_state.clear_sweep()
            return

        # The displacement candle must come AFTER the sweep bar closed
        if c2['time'] < sweep.close_time:
            return

        if bullish:
            has_gap = c1['high'] < c3['low']
            is_displacement = c2['close'] > c2['open']
            is_holding = c3['close'] > c1['high']
            sl_raw = float(c1['low'])
        else:
            has_gap = c1['low'] > c3['high']
            is_displacement = c2['close'] < c2['open']
            is_holding = c3['close'] < c1['low']
            sl_raw = float(c1['high'])

        if has_gap and is_displacement and is_holding:
            logger.info(f"[{self.name}] M1 FVG confirmed ({sweep.direction}).")
            if self.execute_trade(sweep.direction, sl_raw):
                shared_state.clear_sweep()  # consume the setup: one sweep, one trade
                self.cooldown_until = time.time() + self.cooldown_sec


# ==========================================
# 5.5 TRADE MANAGER AGENT - BREAK EVEN
# ==========================================
class TradeManagerAgent(BaseAgent):
    """Moves SL to break-even (plus a small buffer) at N x initial risk.
    Only touches positions opened by this system (matching MAGIC)."""
    banner = "Monitoring open positions for Break-Even triggers..."

    def __init__(self, name, symbol, timeframe, sleep_interval,
                 rr=2.0, be_trigger_r=1.0, be_buffer_points=20):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.rr = rr
        self.be_trigger_r = be_trigger_r
        self.be_buffer_points = be_buffer_points  # covers commission/slippage
        self._risks = {}  # ticket -> initial risk distance (price units)

    def step(self):
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            info = mt5.symbol_info(self.symbol)
        if positions is None or info is None:
            return

        point, digits = info.point, info.digits
        stops_dist = info.trade_stops_level * point
        buffer = self.be_buffer_points * point
        live = set()

        for pos in positions:
            if pos.magic != MAGIC:
                continue
            live.add(pos.ticket)
            if pos.sl == 0.0 or pos.tp == 0.0:
                continue

            is_buy = pos.type == mt5.POSITION_TYPE_BUY

            # Record the initial risk the first time we see the ticket
            risk = self._risks.get(pos.ticket)
            if risk is None:
                sl_on_loss_side = (pos.sl < pos.price_open) if is_buy else (pos.sl > pos.price_open)
                if sl_on_loss_side:
                    risk = abs(pos.price_open - pos.sl)
                else:  # SL already at/after BE (e.g. after a restart): fall back to TP
                    risk = abs(pos.tp - pos.price_open) / self.rr
                self._risks[pos.ticket] = risk

            profit = (pos.price_current - pos.price_open) if is_buy \
                else (pos.price_open - pos.price_current)
            secured = (pos.sl >= pos.price_open) if is_buy else (pos.sl <= pos.price_open)

            if secured or profit < risk * self.be_trigger_r:
                continue

            new_sl = round(pos.price_open + buffer, digits) if is_buy \
                else round(pos.price_open - buffer, digits)

            # New SL must be on the correct side of price and outside the stops level
            if is_buy and new_sl >= pos.price_current:
                continue
            if not is_buy and new_sl <= pos.price_current:
                continue
            if abs(pos.price_current - new_sl) < stops_dist:
                continue

            logger.info(f"[{self.name}] Ticket {pos.ticket} reached {self.be_trigger_r}R. Securing break-even.")
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,  # must be re-sent or it gets cleared
                "magic": MAGIC,
            }
            with mt5_lock:
                result = mt5.order_send(request)

            if result is None:
                logger.info(f"[{self.name}] BE modify returned None: {mt5.last_error()}")
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.info(f"[{self.name}] BE modify failed: retcode={result.retcode} {result.comment}")
            else:
                logger.info(f"[{self.name}] SL moved to {new_sl} on ticket {pos.ticket}.")

        # Forget tickets that are no longer open
        for t in list(self._risks):
            if t not in live:
                del self._risks[t]


# ==========================================
# 6. ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        logger.info(f"MT5 initialization failed: {mt5.last_error()}")
        quit()

    SYMBOL = "XAUUSD.sd"
    if not mt5.symbol_select(SYMBOL, True):
        logger.info(f"Could not select {SYMBOL}: {mt5.last_error()}")
        mt5.shutdown()
        quit()

    RR = 2.0

    macro = MacroAgent("Macro_H4", SYMBOL, mt5.TIMEFRAME_H4, sleep_interval=60)
    intermediate = IntermediateAgent(
        "Intermediate_M15", SYMBOL, mt5.TIMEFRAME_M15, sleep_interval=15,
        allow_shorts=False,      # set True to also trade bearish supply-zone upthrusts
        sweep_ttl_sec=3600,
    )
    micro = MicroAgent(
        "Micro_M1", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2,
        use_dynamic_lot=True, fixed_lot=0.10, risk_pct=1.0,
        rr=RR, deviation=20, cooldown_sec=300,
        min_sl_spread_mult=3.0, max_spread_points=None,  # e.g. 60 to cap spread
    )
    manager = TradeManagerAgent(
        "Manager", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=3,
        rr=RR, be_trigger_r=1.0, be_buffer_points=20,
    )

    agents = [macro, intermediate, micro, manager]
    for a in agents:
        a.start()

    warned = set()
    try:
        logger.info("Multi-agent system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(5)
            for a in agents:
                if a.name in warned:
                    continue
                if not a.is_alive():
                    logger.info(f"!! [{a.name}] thread is DEAD.")
                    warned.add(a.name)
                elif time.time() - a.last_beat > max(60, a.sleep_interval * 10):
                    logger.info(f"!! [{a.name}] has not completed a loop in "
                          f"{time.time() - a.last_beat:.0f}s (stuck MT5 call?).")
                    warned.add(a.name)
    except KeyboardInterrupt:
        logger.info("\nShutting down safely...")
        for a in agents:
            a.stop()
        for a in agents:
            a.join(timeout=10)
        mt5.shutdown()
        logger.info("Shutdown complete.")