import MetaTrader5 as mt5
import pandas as pd
import threading
import time
import math

# ==========================================
# 1. SHARED STATE & LOCKS
# ==========================================
class TradingState:
    def __init__(self):
        self.macro_bias = "NEUTRAL"
        self.macro_poi_active = False
        self.liquidity_swept = False
        self.sweep_direction = None
        
        self.state_lock = threading.Lock()
        self.order_lock = threading.Lock()

    def update_macro(self, bias, in_poi):
        with self.state_lock:
            self.macro_bias = bias
            self.macro_poi_active = in_poi

    def update_sweep(self, swept, direction=None):
        with self.state_lock:
            self.liquidity_swept = swept
            self.sweep_direction = direction

shared_state = TradingState()

# ==========================================
# 2. BASE AGENT
# ==========================================
class BaseAgent(threading.Thread):
    def __init__(self, name, symbol, timeframe, sleep_interval):
        super().__init__()
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.sleep_interval = sleep_interval
        self.running = True

    def fetch_data(self, bars=100):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def stop(self):
        self.running = False

# ==========================================
# 3. MACRO AGENT (D1 / H4) - BIAS & POI
# ==========================================
class MacroAgent(BaseAgent):
    def run(self):
        print(f"[{self.name}] Tracking Institutional Bias...")
        while self.running:
            df = self.fetch_data(bars=250)
            if df is not None:
                # 1. Macro Trend (EMA 50 / 200)
                df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
                df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
                bias = "BULLISH" if df['ema50'].iloc[-1] > df['ema200'].iloc[-1] else "BEARISH"
                
                # 2. Point of Interest (Demand Zone)
                df['atr'] = df['high'] - df['low']
                avg_atr = df['atr'].rolling(14).mean()
                is_expansion = (df['close'] > df['open']) & (df['atr'] > (1.5 * avg_atr))
                
                in_poi = False
                if is_expansion.any():
                    last_exp_idx = df[is_expansion].index[-1]
                    if last_exp_idx > 0:
                        demand_zone_high = df.loc[last_exp_idx - 1, 'high']
                        demand_zone_low = df.loc[last_exp_idx - 1, 'low']
                        current_price = df['close'].iloc[-1]
                        in_poi = demand_zone_low <= current_price <= demand_zone_high

                shared_state.update_macro(bias, in_poi)
            time.sleep(self.sleep_interval)

# ==========================================
# 4. INTERMEDIATE AGENT (H1 / M15) - WYCKOFF SPRING
# ==========================================
class IntermediateAgent(BaseAgent):
    def run(self):
        print(f"[{self.name}] Hunting for Liquidity Sweeps...")
        while self.running:
            with shared_state.state_lock:
                bias = shared_state.macro_bias
                in_poi = shared_state.macro_poi_active

            if in_poi and bias != "NEUTRAL":
                df = self.fetch_data(bars=50)
                if df is not None:
                    swing_low = df['low'].iloc[-40:-5].min()
                    current_low = df['low'].iloc[-1]
                    current_close = df['close'].iloc[-1]
                    current_open = df['open'].iloc[-1]
                    
                    df['vol_ma'] = df['tick_volume'].rolling(20).mean()
                    is_climax_volume = df['tick_volume'].iloc[-1] > (df['vol_ma'].iloc[-1] * 1.5)

                    # Spring: Spikes below low, closes above it, bullish candle, high volume
                    is_bullish_sweep = (
                        current_low < swing_low and 
                        current_close > swing_low and 
                        current_close > current_open and
                        is_climax_volume
                    )

                    if bias == "BULLISH" and is_bullish_sweep:
                        print(f"[{self.name}] Wyckoff Spring Detected!")
                        shared_state.update_sweep(True, "BULLISH_SWEEP")
                    else:
                        shared_state.update_sweep(False, None)
            else:
                shared_state.update_sweep(False, None)
                
            time.sleep(self.sleep_interval)

# ==========================================
# 5. MICRO AGENT (M5 / M1) - FVG & EXECUTION
# ==========================================
class MicroAgent(BaseAgent):
    def __init__(self, name, symbol, timeframe, sleep_interval, use_dynamic_lot, fixed_lot, risk_pct):
        super().__init__(name, symbol, timeframe, sleep_interval)
        self.use_dynamic_lot = use_dynamic_lot
        self.fixed_lot = fixed_lot
        self.risk_pct = risk_pct

    def calculate_lot_size(self, stop_loss_price, entry_price):
        """Calculates dynamic lot size based on account balance and risk %"""
        if not self.use_dynamic_lot:
            return self.fixed_lot

        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(self.symbol)
        
        if not account_info or not symbol_info:
            print(f"[{self.name}] Failed to get account/symbol info. Using fixed lot.")
            return self.fixed_lot

        balance = account_info.balance
        risk_amount = balance * (self.risk_pct / 100)
        
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        
        # Calculate risk per 1 standard lot
        distance_in_points = abs(entry_price - stop_loss_price) / tick_size
        risk_per_lot = distance_in_points * tick_value
        
        if risk_per_lot == 0:
            return symbol_info.volume_min

        # Calculate exact lot and clamp to broker limits
        raw_lot = risk_amount / risk_per_lot
        step = symbol_info.volume_step
        
        lot_size = math.floor(raw_lot / step) * step
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        return round(lot_size, 2)

    def execute_trade(self, action, sl_price):
        tick = mt5.symbol_info_tick(self.symbol)
        entry_price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
        
        lot_size = self.calculate_lot_size(sl_price, entry_price)
        
        # Calculate a 2:1 Reward to Risk Take Profit
        risk_distance = abs(entry_price - sl_price)
        tp_price = entry_price + (risk_distance * 2) if action == mt5.ORDER_TYPE_BUY else entry_price - (risk_distance * 2)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": action,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 999999,
            "comment": "Wyckoff_FVG_Entry",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        with shared_state.order_lock:
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[{self.name}] Trade Failed: {result.comment}")
            else:
                print(f"[{self.name}] Executed {lot_size} lots at {result.price}. SL: {sl_price}, TP: {tp_price}")

    def run(self):
        print(f"[{self.name}] Waiting for Liquidity Sweeps to execute FVGs...")
        while self.running:
            with shared_state.state_lock:
                sweep = shared_state.liquidity_swept
                sweep_dir = shared_state.sweep_direction

            if sweep:
                df = self.fetch_data(bars=5)
                if df is not None:
                    # FVG Logic (3-candle formation)
                    c1_high = df['high'].iloc[-3]
                    c2_open = df['open'].iloc[-2]
                    c2_close = df['close'].iloc[-2]
                    c3_low = df['low'].iloc[-1]

                    has_gap = c1_high < c3_low
                    is_displacement = c2_close > c2_open
                    is_holding = df['close'].iloc[-1] > c1_high

                    if sweep_dir == "BULLISH_SWEEP" and has_gap and is_displacement and is_holding:
                        print(f"[{self.name}] M1 FVG Confirmed!")
                        
                        # Stop Loss goes just below Candle 1 of the FVG
                        stop_loss = df['low'].iloc[-3]
                        self.execute_trade(mt5.ORDER_TYPE_BUY, sl_price=stop_loss)
                        
                        # Reset state and cooldown
                        shared_state.update_sweep(False, None)
                        time.sleep(300) 
                        
            time.sleep(self.sleep_interval)
# ==========================================
# 5.5 TRADE MANAGER AGENT - BREAK EVEN LOGIC
# ==========================================
class TradeManagerAgent(BaseAgent):
    """Monitors open positions and moves SL to break-even at 1:1 RR."""
    def run(self):
        print(f"[{self.name}] Monitoring open positions for Break-Even triggers...")
        while self.running:
            # Fetch all open positions for the symbol
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions:
                for pos in positions:
                    # Skip trades that do not have a SL or TP set
                    if pos.tp == 0.0 or pos.sl == 0.0:
                        continue 

                    # Back-calculate the initial 1R (risk) distance.
                    # Since MicroAgent sets TP at 2R, 1R is half the distance to TP.
                    risk_distance = abs(pos.tp - pos.price_open) / 2
                    
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        current_profit = pos.price_current - pos.price_open
                        is_1_to_1_reached = current_profit >= risk_distance
                        is_sl_at_be = pos.sl >= pos.price_open
                    else: # mt5.ORDER_TYPE_SELL
                        current_profit = pos.price_open - pos.price_current
                        is_1_to_1_reached = current_profit >= risk_distance
                        is_sl_at_be = pos.sl <= pos.price_open

                    # If 1:1 RR is hit and SL is not already moved
                    if is_1_to_1_reached and not is_sl_at_be:
                        print(f"[{self.name}] Trade {pos.ticket} reached 1:1 RR. Securing Break-Even.")
                        
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": pos.symbol,
                            "position": pos.ticket,
                            "sl": pos.price_open, # Move SL to exact entry price
                            "tp": pos.tp          # You MUST include the existing TP to keep it
                        }

                        # Use the shared lock to prevent API collisions with the Micro Agent
                        with shared_state.order_lock:
                            result = mt5.order_send(request)
                            if result.retcode != mt5.TRADE_RETCODE_DONE:
                                print(f"[{self.name}] BE Modification Failed: {result.comment}")
                            else:
                                print(f"[{self.name}] Stop Loss moved to Entry ({pos.price_open}). Trade is risk-free.")

            # Run this check every few seconds
            time.sleep(self.sleep_interval)

# ==========================================
# 6. ORCHESTRATOR
# ==========================================
# ==========================================
# 6. ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize():
        print("MT5 initialization failed.")
        quit()

    SYMBOL = "XAUUSD.sd"
    mt5.symbol_select(SYMBOL, True)

    # 1. Instantiate the Agents
    macro = MacroAgent("Macro_H4", SYMBOL, mt5.TIMEFRAME_H4, sleep_interval=60)
    intermediate = IntermediateAgent("Intermediate_M15", SYMBOL, mt5.TIMEFRAME_M15, sleep_interval=15)
    micro = MicroAgent("Micro_M1", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=2, use_dynamic_lot=True, fixed_lot=0.10, risk_pct=1.0)
    
    # Instantiate the Trade Manager (checking positions every 3 seconds)
    manager = TradeManagerAgent("Manager", SYMBOL, mt5.TIMEFRAME_M1, sleep_interval=3)

    # 2. Start the Threads
    macro.start()
    intermediate.start()
    micro.start()
    manager.start()

    try:
        print("Multi-Agent System with Break-Even Management running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down safely...")
        macro.stop()
        intermediate.stop()
        micro.stop()
        manager.stop()
        
        macro.join()
        intermediate.join()
        micro.join()
        manager.join()
        
        mt5.shutdown()
        print("Shutdown complete.")