import math
import threading
import time
import traceback
import os
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# ==========================================
# 0. CONSTANTS, LOCKS & LOGGER
# ==========================================
MAGIC = 999999
BULL_SWEEP = "BULLISH_SWEEP"
BEAR_SWEEP = "BEARISH_SWEEP"

# Global lock for MT5 calls to prevent C-backend crashes
mt5_lock = threading.RLock()

# Logger Setup
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_handler = RotatingFileHandler(os.path.join(LOG_DIR, "wyckoff_bot.log"), maxBytes=5*1024*1024, backupCount=3)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)

logger = logging.getLogger("WyckoffBot")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

cli_handler = logging.StreamHandler()
cli_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(cli_handler)

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

class TradingState:
    def __init__(self):
        self.macro_bias = "NEUTRAL"
        self.macro_poi_active = False
        self.macro_poi_levels = {"high": 0.0, "low": 0.0}
        self.macro_ts = 0.0
        
        self._sweep = None
        self.m15_levels = {"swing_low": 0.0, "swing_high": 0.0, "vol_ratio": 0.0}
        
        self.agent_statuses = {
            "Macro_H4": "Initializing...",
            "Intermediate_M15": "Initializing...",
            "Swing_H1": "Initializing...",
            "Micro_M1": "Initializing...",
            "Manager": "Initializing..."
        }
        self.state_lock = threading.Lock()

    def update_agent_status(self, agent_name, status):
        with self.state_lock:
            self.agent_statuses[agent_name] = status

    def update_macro(self, bias, in_poi, poi_high=0.0, poi_low=0.0):
        with self.state_lock:
            self.macro_bias = bias
            self.macro_poi_active = in_poi
            self.macro_poi_levels = {"high": poi_high, "low": poi_low}
            self.macro_ts = time.time()

    def get_macro(self, max_age=600):
        with self.state_lock:
            fresh = (time.time() - self.macro_ts) <= max_age
            return self.macro_bias, self.macro_poi_active, fresh

    def update_m15_telemetry(self, swing_low, swing_high, vol_ratio):
        with self.state_lock:
            self.m15_levels = {"swing_low": swing_low, "swing_high": swing_high, "vol_ratio": vol_ratio}

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
            try:
                self.step()
            except Exception as e:
                logger.error(f"[{self.name}] Error in step(): {e}", exc_info=True)
            self._stop_event.wait(self.sleep_interval)

    def stop(self):
        self._stop_event.set()

# ==========================================
# 3. MACRO AGENT (H4)
# ==========================================
class MacroAgent(BaseAgent):
    EXPANSION_MULT = 1.5
    ATR_PERIOD = 14
    MAX_ZONE_AGE = 100

    def step(self):
        df = self.fetch_data(bars=300)
        if df is None or len(df) < 210:
            shared_state.update_agent_status(self.name, "Awaiting Data...")
            return

        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        ema200 = df['close'].ewm(span=200, adjust=False).mean()
        bias = "BULLISH" if ema50.iloc[-1] > ema200.iloc[-1] else "BEARISH"

        prev_close = df['close'].shift(1)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()),
        )
        atr = tr.rolling(self.ATR_PERIOD).mean().shift(1)
        big = tr > self.EXPANSION_MULT * atr

        bullish = bias == "BULLISH"
        direction = (df['close'] > df['open']) if bullish else (df['close'] < df['open'])
        idx = np.flatnonzero((direction & big).to_numpy())
        idx = idx[idx >= 1]
        
        in_poi, z_high, z_low = False, 0.0, 0.0

        if len(idx) > 0:
            i = int(idx[-1])
            if (len(df) - 1) - i <= self.MAX_ZONE_AGE:
                z_high = df['high'].iloc[i - 1]
                z_low = df['low'].iloc[i - 1]
                
                after = df['close'].iloc[i + 1:]
                invalid = ((after < z_low).any() if bullish else (after > z_high).any())
                
                if not invalid:
                    price = df['close'].iloc[-1]
                    in_poi = bool(z_low <= price <= z_high)

        shared_state.update_macro(bias, in_poi, poi_high=z_high, poi_low=z_low)
        shared_state.update_agent_status(self.name, f"Bias: {bias} | In POI: {in_poi}")

# ==========================================
# 4. INTERMEDIATE AGENT (M15)
# ==========================================
class IntermediateAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, allow_shorts=False, sweep_ttl_sec=3600):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.allow_shorts = allow_shorts
        self.sweep_ttl_sec = sweep_ttl_sec
        self.last_bar_time = None

    def step(self):
        bias, in_poi, fresh = shared_state.get_macro()
        if not (fresh and in_poi):
            shared_state.update_agent_status(self.name, "Idle (Waiting for Macro POI)")
            return
            
        if bias == "BEARISH" and not self.allow_shorts:
            shared_state.update_agent_status(self.name, "Idle (Shorts Disabled)")
            return
            
        if shared_state.get_sweep() is not None:
            shared_state.update_agent_status(self.name, "Sweep Latched. Waiting for execution.")
            return

        df = self.fetch_data(bars=60)
        if df is None or len(df) < 50:
            return

        bar = df.iloc[-1]
        if bar['time'] == self.last_bar_time:
            return
        self.last_bar_time = bar['time']

        vol_ma = df['tick_volume'].iloc[-21:-1].mean()
        climax = bar['tick_volume'] > (vol_ma * 1.5)
        vol_ratio = bar['tick_volume'] / vol_ma if vol_ma > 0 else 1.0

        level, hit, direction, extreme = 0.0, False, None, 0.0

        if bias == "BULLISH":
            level = df['low'].iloc[-45:-5].min()
            hit = (bar['low'] < level and bar['close'] > level and bar['close'] > bar['open'] and climax)
            direction, extreme = BULL_SWEEP, float(bar['low'])
            shared_state.update_m15_telemetry(level, 0.0, vol_ratio)
        else:
            level = df['high'].iloc[-45:-5].max()
            hit = (bar['high'] > level and bar['close'] < level and bar['close'] < bar['open'] and climax)
            direction, extreme = BEAR_SWEEP, float(bar['high'])
            shared_state.update_m15_telemetry(0.0, level, vol_ratio)

        if hit:
            bar_delta = df['time'].iloc[-1] - df['time'].iloc[-2]
            shared_state.set_sweep(Sweep(
                direction=direction, level=float(level), extreme=extreme,
                bar_time=bar['time'], close_time=bar['time'] + bar_delta,
                expires_at=time.time() + self.sweep_ttl_sec,
            ))
            logger.info(f"[{self.name}] {direction} latched at {bar['time']} (level {level:.2f})")
            shared_state.update_agent_status(self.name, f"LATCHED: {direction}")
        else:
            shared_state.update_agent_status(self.name, f"Scanning | Vol: {vol_ratio:.2f}x")

# ==========================================
# 5. MICRO / SWING AGENT LOGIC
# ==========================================
class ExecutionAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, use_dynamic_lot, fixed_lot, risk_pct, rr, deviation, cooldown_sec, min_sl_spread_mult):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.use_dynamic_lot = use_dynamic_lot
        self.fixed_lot = fixed_lot
        self.risk_pct = risk_pct
        self.rr = rr
        self.deviation = deviation
        self.cooldown_sec = cooldown_sec
        self.min_sl_spread_mult = min_sl_spread_mult
        self.cooldown_until = 0.0
        self.last_bar_time = None

    def has_open_position(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return True 
        return any(p.magic == MAGIC for p in positions)

    def calculate_lot_size(self, info, entry, sl):
        step = info.volume_step
        if not self.use_dynamic_lot:
            lot = math.floor(self.fixed_lot / step + 1e-9) * step
            return max(info.volume_min, min(lot, info.volume_max))

        acct = mt5.account_info()
        if not acct: return None

        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value_loss or info.trade_tick_value
        risk_per_lot = (abs(entry - sl) / tick_size) * tick_value
        
        if risk_per_lot <= 0: return None
        raw_lot = (acct.balance * (self.risk_pct / 100.0)) / risk_per_lot
        lot = math.floor(raw_lot / step + 1e-9) * step
        
        if lot < info.volume_min: return None
        return min(lot, info.volume_max)

    def execute_trade(self, direction, sl_raw):
        is_buy = direction == BULL_SWEEP
        with mt5_lock:
            if self.has_open_position(): return False
            info, tick = mt5.symbol_info(self.symbol), mt5.symbol_info_tick(self.symbol)
            if not info or not tick: return False

            spread = tick.ask - tick.bid
            entry = tick.ask if is_buy else tick.bid
            sl = sl_raw - spread if is_buy else sl_raw + spread

            risk = abs(entry - sl)
            min_dist = max(info.trade_stops_level * info.point, spread * self.min_sl_spread_mult)
            if risk < min_dist: return False

            tp = entry + risk * self.rr if is_buy else entry - risk * self.rr
            lot = self.calculate_lot_size(info, entry, sl)
            if lot is None: return False

            req = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": round(entry, info.digits), "sl": round(sl, info.digits), "tp": round(tp, info.digits),
                "deviation": self.deviation, "magic": MAGIC, "comment": "Wyckoff_Bot",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
            }
            res = mt5.order_send(req)

        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[{self.name}] Executed {lot} lots at {res.price}. SL: {req['sl']}")
            return True
        logger.warning(f"[{self.name}] Trade rejected. Code: {res.retcode if res else 'None'}")
        return False

class SwingAgent(ExecutionAgent):
    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        bias, in_poi, fresh = shared_state.get_macro()
        if not (fresh and in_poi) or bias not in ("BULLISH", "BEARISH"):
            shared_state.update_agent_status(self.name, "Idle (Waiting for Macro POI)")
            return

        shared_state.update_agent_status(self.name, f"Hunting H1 FVG ({bias})")
        with mt5_lock:
            if self.has_open_position(): return

        df = self.fetch_data(bars=10)
        if df is None or len(df) < 5: return

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c3['time'] == self.last_bar_time: return 
        self.last_bar_time = c3['time']

        bullish = bias == "BULLISH"
        if bullish:
            has_gap, is_disp, is_hold = c1['high'] < c3['low'], c2['close'] > c2['open'], c3['close'] > c1['high']
            direction, sl_raw = BULL_SWEEP, float(c1['low'])
        else:
            has_gap, is_disp, is_hold = c1['low'] > c3['high'], c2['close'] < c2['open'], c3['close'] < c1['low']
            direction, sl_raw = BEAR_SWEEP, float(c1['high'])

        if has_gap and is_disp and is_hold:
            if self.execute_trade(direction, sl_raw):
                self.cooldown_until = time.time() + self.cooldown_sec

class MicroAgent(ExecutionAgent):
    def step(self):
        if time.time() < self.cooldown_until:
            shared_state.update_agent_status(self.name, f"Cooldown ({int(self.cooldown_until - time.time())}s)")
            return

        sweep = shared_state.get_sweep()
        if sweep is None:
            shared_state.update_agent_status(self.name, "Idle (Waiting for Sweep)")
            return

        shared_state.update_agent_status(self.name, f"Hunting M1 FVG ({sweep.direction})")
        with mt5_lock:
            if self.has_open_position(): return

        df = self.fetch_data(bars=10)
        if df is None or len(df) < 5: return

        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c3['time'] == self.last_bar_time: return
        self.last_bar_time = c3['time']

        bullish = sweep.direction == BULL_SWEEP
        if (bullish and c3['close'] < sweep.extreme) or (not bullish and c3['close'] > sweep.extreme):
            shared_state.clear_sweep()
            return

        if c2['time'] < sweep.close_time: return

        if bullish:
            has_gap, is_disp, is_hold = c1['high'] < c3['low'], c2['close'] > c2['open'], c3['close'] > c1['high']
            sl_raw = float(c1['low'])
        else:
            has_gap, is_disp, is_hold = c1['low'] > c3['high'], c2['close'] < c2['open'], c3['close'] < c1['low']
            sl_raw = float(c1['high'])

        if has_gap and is_disp and is_hold:
            if self.execute_trade(sweep.direction, sl_raw):
                shared_state.clear_sweep()
                self.cooldown_until = time.time() + self.cooldown_sec

# ==========================================
# 6. TRADE MANAGER AGENT
# ==========================================
class TradeManagerAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, rr, be_buffer_points):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.rr = rr
        self.be_buffer_points = be_buffer_points
        self._risks = {}

    def step(self):
        shared_state.update_agent_status(self.name, "Monitoring Break-Even Triggers")
        with mt5_lock:
            positions = mt5.positions_get(symbol=self.symbol)
            info = mt5.symbol_info(self.symbol)
            
        if not positions or not info: return

        point, digits = info.point, info.digits
        buffer = self.be_buffer_points * point
        live = set()

        for pos in positions:
            if pos.magic != MAGIC or pos.sl == 0.0 or pos.tp == 0.0: continue
            live.add(pos.ticket)

            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            if pos.ticket not in self._risks:
                if (is_buy and pos.sl < pos.price_open) or (not is_buy and pos.sl > pos.price_open):
                    self._risks[pos.ticket] = abs(pos.price_open - pos.sl)
                else:
                    self._risks[pos.ticket] = abs(pos.tp - pos.price_open) / self.rr

            risk = self._risks[pos.ticket]
            profit = (pos.price_current - pos.price_open) if is_buy else (pos.price_open - pos.price_current)
            secured = (pos.sl >= pos.price_open) if is_buy else (pos.sl <= pos.price_open)

            if secured or profit < risk: continue

            new_sl = round(pos.price_open + buffer, digits) if is_buy else round(pos.price_open - buffer, digits)
            
            if (is_buy and new_sl >= pos.price_current) or (not is_buy and new_sl <= pos.price_current): continue
            if abs(pos.price_current - new_sl) < (info.trade_stops_level * point): continue

            req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol, "position": pos.ticket, "sl": new_sl, "tp": pos.tp, "magic": MAGIC}
            with mt5_lock:
                res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[{self.name}] Ticket {pos.ticket} secured at Break-Even.")

        self._risks = {t: v for t, v in self._risks.items() if t in live}

# ==========================================
# 7. WEB DASHBOARD API
# ==========================================
app = FastAPI(title="Wyckoff Bot Terminal")

@app.get("/api/state")
def get_system_state():
    with shared_state.state_lock:
        bias = shared_state.macro_bias
        in_poi = shared_state.macro_poi_active
        poi_levels = shared_state.macro_poi_levels
        m15_levels = shared_state.m15_levels
        agent_statuses = dict(shared_state.agent_statuses)
        fresh = (time.time() - shared_state.macro_ts) <= 600

    sweep = shared_state.get_sweep()
    sweep_data = None
    if sweep:
        sweep_data = {"direction": sweep.direction, "level": sweep.level, "expires_in_sec": max(0, round(sweep.expires_at - time.time()))}

    open_trades, account_data, market_data = [], {}, {}
    with mt5_lock:
        acct = mt5.account_info()
        if acct: account_data = {"balance": acct.balance, "equity": acct.equity, "profit": acct.profit}
        info = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        if info and tick: market_data = {"symbol": info.name, "bid": tick.bid, "ask": tick.ask, "spread": round((tick.ask - tick.bid) / info.point, 1)}
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions:
            for p in positions:
                if p.magic == MAGIC: open_trades.append({"ticket": p.ticket, "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL", "vol": p.volume, "profit": p.profit})

    return {"market": market_data, "account": account_data, "macro": {"bias": bias, "in_poi": in_poi, "levels": poi_levels, "fresh": fresh}, "m15": m15_levels, "sweep": sweep_data, "agents": agent_statuses, "trades": open_trades}

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """
    <html><head><title>Agent Dashboard</title><meta http-equiv="refresh" content="3"><style>body{font-family:sans-serif;background:#0c0d0e;color:#e1e3e6;padding:20px} .card{background:#16181b;padding:15px;margin:10px 0;border-radius:8px;} .bull{color:#3fb950;} .bear{color:#f85149;}</style></head>
    <body><h2>Wyckoff Command Terminal</h2><div class="card" id="data">Loading...</div>
    <script>
        async function load() {
            const r = await fetch('/api/state'); const d = await r.json();
            let h = `<p>Symbol: ${d.market.symbol || '--'} | Spread: ${d.market.spread || '--'} pts</p>`;
            h += `<p>Balance: $${d.account.balance ? d.account.balance.toFixed(2) : '--'} | PnL: $${d.account.profit ? d.account.profit.toFixed(2) : '--'}</p><hr/>`;
            for (const [k, v] of Object.entries(d.agents)) h += `<p>${k}: ${v}</p>`;
            h += `<hr/><p>H4 Bias: <strong class="${d.macro.bias==='BULLISH'?'bull':'bear'}">${d.macro.bias}</strong> | In POI: ${d.macro.in_poi}</p>`;
            h += `<p>Active M15 Sweep: ${d.sweep ? d.sweep.direction + ' (' + d.sweep.expires_in_sec + 's)' : 'None'}</p>`;
            document.getElementById('data').innerHTML = h;
        } load();
    </script></body></html>
    """

def run_web_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="critical")

# ==========================================
# 8. ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        quit()

    SYMBOL = "XAUUSD.sd"
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Could not select {SYMBOL}")
        mt5.shutdown()
        quit()

    RR = 2.0
    macro = MacroAgent("Macro_H4", SYMBOL, mt5.TIMEFRAME_H4, sleep_interval=60)
    inter = IntermediateAgent("Intermediate_M15", SYMBOL, mt5.TIMEFRAME_M15, sleep_interval=15, allow_shorts=True, sweep_ttl_sec=3600)
    swing = SwingAgent("Swing_H1", SYMBOL, mt5.TIMEFRAME_H1, sleep_interval=15, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=1.0, rr=RR, deviation=20, cooldown_sec=3600, min_sl_spread_mult=3.0)
    micro = MicroAgent("Micro_M1", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2, use_dynamic_lot=True, fixed_lot=0.1, risk_pct=1.0, rr=RR, deviation=20, cooldown_sec=300, min_sl_spread_mult=3.0)
    manager = TradeManagerAgent("Manager", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=3, rr=RR, be_buffer_points=20)

    agents = [macro, inter, swing, micro, manager]
    for a in agents: a.start()

    threading.Thread(target=run_web_server, daemon=True).start()
    
    try:
        logger.info("System running. Access Dashboard at http://127.0.0.1:8000. Press Ctrl+C to stop.")
        while True:
            time.sleep(5)
            for a in agents:
                if not a.is_alive(): logger.error(f"!! {a.name} thread DEAD !!")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for a in agents: a.stop()
        for a in agents: a.join(timeout=10)
        mt5.shutdown()