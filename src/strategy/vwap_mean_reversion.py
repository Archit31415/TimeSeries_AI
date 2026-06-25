import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VWAPMeanReversionStrategy:
    """
    Dynamic VWAP Mean Reversion Strategy, gated by HMM Regime Detection.
    Executes trades only in 'Chop' regimes and sizes positions inversely to predicted volatility.
    """
    def __init__(self, z_score_threshold: float = 2.0, base_risk_fraction: float = 0.05, smoothing_alpha: float = 0.15):
        """
        Args:
            z_score_threshold: The number of standard deviations (σ) from VWAP to trigger a trade.
            base_risk_fraction: The base percentage of capital to risk per trade (e.g., 0.05 = 5%).
            smoothing_alpha: Exponential moving average smoothing parameter for regime probabilities.
        """
        self.z_score_threshold = z_score_threshold
        self.base_risk_fraction = base_risk_fraction
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_probs = None
        self.active_regime = None
        
        logging.info(f"Initialized VWAP Mean Reversion Strategy (Threshold: {z_score_threshold}σ, Base Risk: {base_risk_fraction*100}%, Smoothing Alpha: {smoothing_alpha})")

    def execute(self, current_price: float, vwap: float, predicted_vol: float, 
                regime_probs: dict, available_capital: float, current_holdings: float) -> dict:
        """
        The core state-machine execution function called by the C++ IPC Hub.
        
        Args:
            current_price: The latest asset price.
            vwap: The current anchored Volume-Weighted Average Price.
            predicted_vol: The LSTM's forecasted standard deviation (σ).
            regime_probs: Dictionary of probabilities from the HMM-LSTM.
            available_capital: Total USDT available to trade.
            current_holdings: Total ETH currently held (positive for long, negative for short).
            
        Returns:
            dict: Expected by the backtest/C++ engine in the format:
                  {"buy": X, "sell": Y, "exec_price": P, "is_maker": bool}
        """

        # 1. Translate Daily Volatility down to a 5-Minute Volatility frame
        # (Divide by the square root of 288 bars)
        pred_5min_vol = predicted_vol / math.sqrt(288)

        # 2. Correct the Band Scaling Formula: scale the VWAP asset price geometrically, not linearly
        upper_band = vwap * (1.0 + self.z_score_threshold * pred_5min_vol)
        lower_band = vwap * (1.0 - self.z_score_threshold * pred_5min_vol)
        
        # 3. Regime Hysteresis (Smoothing via EMA filter)
        if self.smoothed_probs is None:
            self.smoothed_probs = {
                "State_0_Downtrend": regime_probs.get("State_0_Downtrend", 0.0),
                "State_1_Uptrend": regime_probs.get("State_1_Uptrend", 0.0),
                "State_2_Chop": regime_probs.get("State_2_Chop", 0.0)
            }
        else:
            for key in ["State_0_Downtrend", "State_1_Uptrend", "State_2_Chop"]:
                raw_val = regime_probs.get(key, 0.0)
                prev_val = self.smoothed_probs.get(key, raw_val)
                self.smoothed_probs[key] = self.smoothing_alpha * raw_val + (1.0 - self.smoothing_alpha) * prev_val
        
        prob_downtrend = self.smoothed_probs["State_0_Downtrend"]
        prob_uptrend = self.smoothed_probs["State_1_Uptrend"]
        prob_chop = self.smoothed_probs["State_2_Chop"]

        # Enforce state hysteresis gate (0.50 threshold to trigger a regime change)
        if self.active_regime is None:
            # First run: select dominant regime
            self.active_regime = max(
                [("Downtrend", prob_downtrend), ("Uptrend", prob_uptrend), ("Chop", prob_chop)],
                key=lambda x: x[1]
            )[0]
        else:
            if prob_chop > 0.50:
                self.active_regime = "Chop"
            elif prob_uptrend > 0.50:
                self.active_regime = "Uptrend"
            elif prob_downtrend > 0.50:
                self.active_regime = "Downtrend"

        orders = {"buy": 0.0, "sell": 0.0, "exec_price": current_price, "is_maker": False}

        # ==========================================
        # STRATEGY DISPATCH BY ACTIVE REGIME
        # ==========================================
        if self.active_regime == "Chop":
            # --- CHOP: Mean Reversion ---
            # Exit long position: if price >= vwap, exit via limit order at vwap
            if current_holdings > 0:
                if current_price >= vwap:
                    logging.info(f"MR Exit: Price {current_price:.2f} >= VWAP {vwap:.2f}. Selling to close long.")
                    orders["sell"] = current_holdings
                    orders["exec_price"] = current_price
                    orders["is_maker"] = True
            
            # Exit short position: if we are holding a short position, flatten it at current price (market/taker)
            elif current_holdings < 0:
                logging.info(f"MR Flatten: Active regime is Chop but holding short. Buying to cover.")
                orders["buy"] = abs(current_holdings)
                orders["exec_price"] = current_price
                orders["is_maker"] = False
            
            # Entry long: buy limit order at lower_band if price pierces lower_band
            else:
                if current_price < lower_band:
                    logging.info(f"MR Entry: Price {current_price:.2f} < Lower Band {lower_band:.2f}. Placing Limit Buy.")
                    confidence_scaler = (prob_chop - 0.5) * 2.0
                    vol_scaler = 0.01 / max(predicted_vol, 0.001)
                    target_allocation = available_capital * self.base_risk_fraction * confidence_scaler * vol_scaler
                    target_allocation = min(target_allocation, available_capital * 0.25)
                    
                    shares_to_buy = target_allocation / lower_band
                    shares_to_buy = math.floor(shares_to_buy * 10000) / 10000.0
                    
                    orders["buy"] = shares_to_buy
                    orders["exec_price"] = current_price
                    orders["is_maker"] = True

        elif self.active_regime == "Uptrend":
            # --- UPTREND: Momentum Long ---
            # Exit short position: if holding short in uptrend, flatten immediately (taker)
            if current_holdings < 0:
                logging.info(f"Uptrend Flatten: Holding short in Uptrend regime. Buying to cover.")
                orders["buy"] = abs(current_holdings)
                orders["exec_price"] = current_price
                orders["is_maker"] = False
            
            # Exit long position: stop loss if price falls below VWAP (limit sell at vwap) or if regime shifts (handled below)
            elif current_holdings > 0:
                if current_price < vwap:
                    logging.info(f"Uptrend Stop Loss: Price {current_price:.2f} < VWAP {vwap:.2f}. Selling to close long.")
                    orders["sell"] = current_holdings
                    orders["exec_price"] = current_price
                    orders["is_maker"] = True
            
            # Entry long: buy breakout if price crosses above upper_band (limit buy at upper_band)
            else:
                if current_price > upper_band:
                    logging.info(f"Uptrend Entry: Breakout detected! Price {current_price:.2f} > Upper Band {upper_band:.2f}. Placing Limit Buy.")
                    confidence_scaler = (prob_uptrend - 0.5) * 2.0
                    vol_scaler = 0.01 / max(predicted_vol, 0.001)
                    target_allocation = available_capital * self.base_risk_fraction * confidence_scaler * vol_scaler
                    target_allocation = min(target_allocation, available_capital * 0.25)
                    
                    shares_to_buy = target_allocation / upper_band
                    shares_to_buy = math.floor(shares_to_buy * 10000) / 10000.0
                    
                    orders["buy"] = shares_to_buy
                    orders["exec_price"] = current_price
                    orders["is_maker"] = True

        elif self.active_regime == "Downtrend":
            # --- DOWNTREND: Momentum Short ---
            # Exit long position: if holding long in downtrend, flatten immediately (taker)
            if current_holdings > 0:
                logging.info(f"Downtrend Flatten: Holding long in Downtrend regime. Selling to close.")
                orders["sell"] = current_holdings
                orders["exec_price"] = current_price
                orders["is_maker"] = False
            
            # Exit short position: stop loss if price rises above VWAP (limit buy cover at vwap) or if regime shifts
            elif current_holdings < 0:
                if current_price > vwap:
                    logging.info(f"Downtrend Stop Loss: Price {current_price:.2f} > VWAP {vwap:.2f}. Buying to cover short.")
                    orders["buy"] = abs(current_holdings)
                    orders["exec_price"] = current_price
                    orders["is_maker"] = True
            
            # Entry short: sell breakdown if price crosses below lower_band (limit sell at lower_band)
            else:
                if current_price < lower_band:
                    logging.info(f"Downtrend Entry: Breakdown detected! Price {current_price:.2f} < Lower Band {lower_band:.2f}. Placing Limit Sell.")
                    confidence_scaler = (prob_downtrend - 0.5) * 2.0
                    vol_scaler = 0.01 / max(predicted_vol, 0.001)
                    target_allocation = available_capital * self.base_risk_fraction * confidence_scaler * vol_scaler
                    target_allocation = min(target_allocation, available_capital * 0.25)
                    
                    shares_to_sell = target_allocation / lower_band
                    shares_to_sell = math.floor(shares_to_sell * 10000) / 10000.0
                    
                    orders["sell"] = shares_to_sell
                    orders["exec_price"] = current_price
                    orders["is_maker"] = True

        return orders

# --- Module Test & Sanity Check ---
if __name__ == "__main__":
    print("--- Testing VWAP Mean Reversion Logic ---")
    strategy = VWAPMeanReversionStrategy(z_score_threshold=2.0)
    
    # Simulated inputs from the Inference Engine and C++ Hub
    capital = 10000.0   # $10,000 available
    holdings = 0.0      # Holding 0 ETH
    current_px = 3450.0 # ETH Price
    vwap_px = 3500.0    # VWAP is higher (Price is dropping)
    pred_vol = 0.006    # 0.6% predicted volatility for the next 30 mins
    
    print("\nTest 1: Downtrend Regime (Should return 0/0 or sell if holding)")
    regime_trend = {"State_0_Downtrend": 0.85, "State_1_Uptrend": 0.05, "State_2_Chop": 0.10}
    res_1 = strategy.execute(current_px, vwap_px, pred_vol, regime_trend, capital, holdings)
    print(f"Result: {res_1}")

    print("\nTest 2: Chop Regime, Price inside bands (Should return 0/0)")
    strategy.smoothed_probs = None
    strategy.active_regime = None
    regime_chop = {"State_0_Downtrend": 0.15, "State_1_Uptrend": 0.15, "State_2_Chop": 0.70}
    # Lower band = 3500 * (1 - 2 * (0.006 / sqrt(288))) = 3500 * (1 - 2 * 0.0003535) = 3497.5
    # Price 3499.0 is inside the bands.
    res_2 = strategy.execute(3499.0, vwap_px, pred_vol, regime_chop, capital, holdings)
    print(f"Result: {res_2}")

    print("\nTest 3: Chop Regime, Price pierces lower band (Should trigger BUY)")
    # Price is 3450, lower band is 3497.5 -> Oversold!
    res_3 = strategy.execute(current_px, vwap_px, pred_vol, regime_chop, capital, holdings)
    print(f"Result: {res_3}")
    
    print("\nTest 4: Chop Regime, Holding ETH, Price reverts to VWAP (Should trigger SELL)")
    # Price has recovered to 3505, which is > VWAP (3500)
    holdings = res_3["buy"] # Assuming the C++ hub executed our previous buy
    res_4 = strategy.execute(3505.0, vwap_px, pred_vol, regime_chop, capital, holdings)
    print(f"Result: {res_4}")