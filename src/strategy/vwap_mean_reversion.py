import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VWAPMeanReversionStrategy:
    """
    Dynamic VWAP Mean Reversion Strategy, gated by HMM Regime Detection.
    Executes trades only in 'Chop' regimes and sizes positions inversely to predicted volatility.
    """
    def __init__(self, z_score_threshold: float = 2.0, base_risk_fraction: float = 0.05):
        """
        Args:
            z_score_threshold: The number of standard deviations (σ) from VWAP to trigger a trade.
            base_risk_fraction: The base percentage of capital to risk per trade (e.g., 0.05 = 5%).
        """
        self.z_score_threshold = z_score_threshold
        self.base_risk_fraction = base_risk_fraction
        
        logging.info(f"Initialized VWAP Mean Reversion Strategy (Threshold: {z_score_threshold}σ, Base Risk: {base_risk_fraction*100}%)")

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
            current_holdings: Total ETH currently held.
            
        Returns:
            dict: Expected by the C++ engine in the format {"buy": X, "sell": Y}
        """
        # 1. Define the dynamic volatility bands
        upper_band = vwap + (self.z_score_threshold * predicted_vol * current_price)
        lower_band = vwap - (self.z_score_threshold * predicted_vol * current_price)
        
        # 2. Extract Regime Confidences
        prob_chop = regime_probs.get("State_2_Chop", 0.0)
        prob_uptrend = regime_probs.get("State_1_Uptrend", 0.0)
        prob_downtrend = regime_probs.get("State_0_Downtrend", 0.0)
        
        orders = {"buy": 0.0, "sell": 0.0}

        # ==========================================
        # SCENARIO A: TRENDING REGIME (DANGER)
        # ==========================================
        # Mean reversion strategies are systematically destroyed during strong trends.
        # If the model is confident we are trending, we flatten the portfolio to protect capital.
        if prob_uptrend > 0.50 or prob_downtrend > 0.50:
            if current_holdings > 0:
                logging.info(f"Trend Regime Detected (Up: {prob_uptrend:.2f}, Down: {prob_downtrend:.2f}). Flattening portfolio.")
                orders["sell"] = current_holdings
            return orders

        # ==========================================
        # SCENARIO B: CHOP REGIME (TRADE ZONE)
        # ==========================================
        if prob_chop > 0.50:
            
            # --- EXIT LOGIC ---
            # If we are holding a long position and price has reverted back to the VWAP, secure profits.
            if current_holdings > 0 and current_price >= vwap:
                logging.info(f"Price reverted to VWAP ({current_price:.2f} >= {vwap:.2f}). Securing profits.")
                orders["sell"] = current_holdings
                return orders

            # --- ENTRY LOGIC ---
            # If price pierces the lower dynamic band, it is statistically oversold.
            if current_price < lower_band and current_holdings == 0:
                logging.info(f"Oversold anomaly detected: Price {current_price:.2f} < Lower Band {lower_band:.2f}")
                
                # Position Sizing: Fractional Kelly / Volatility Targeting
                # We scale the allocation based on the model's confidence in the 'Chop' state.
                # Example math based on your spec: allocation = capital * (Probability - 0.5) * 2
                confidence_scaler = (prob_chop - 0.5) * 2.0  
                
                # We also scale inversely by volatility: higher predicted vol = smaller position size
                # This ensures consistent risk (volatility targeting).
                vol_scaler = 0.01 / max(predicted_vol, 0.001)  # Normalize against a 1% standard deviation
                
                # Final capital allocation
                target_allocation = available_capital * self.base_risk_fraction * confidence_scaler * vol_scaler
                
                # Cap the allocation to prevent over-leveraging
                target_allocation = min(target_allocation, available_capital * 0.25)
                
                shares_to_buy = target_allocation / current_price
                
                # Floor the shares to the exchange's lot size (e.g., 4 decimal places for ETH)
                shares_to_buy = math.floor(shares_to_buy * 10000) / 10000.0
                
                orders["buy"] = shares_to_buy
                logging.info(f"Executing BUY: {shares_to_buy} ETH at {current_price:.2f}. Total Allocation: ${target_allocation:.2f}")

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
    regime_chop = {"State_0_Downtrend": 0.15, "State_1_Uptrend": 0.15, "State_2_Chop": 0.70}
    # Lower band = 3500 - (2 * 0.006 * 3450) = 3500 - 41.4 = 3458.6
    # Price (3450) is below 3458.6, so it IS outside the bands. Let's adjust price to test "inside bands"
    res_2 = strategy.execute(3480.0, vwap_px, pred_vol, regime_chop, capital, holdings)
    print(f"Result: {res_2}")

    print("\nTest 3: Chop Regime, Price pierces lower band (Should trigger BUY)")
    # Price is 3450, lower band is 3458.6 -> Oversold!
    res_3 = strategy.execute(current_px, vwap_px, pred_vol, regime_chop, capital, holdings)
    print(f"Result: {res_3}")
    
    print("\nTest 4: Chop Regime, Holding ETH, Price reverts to VWAP (Should trigger SELL)")
    # Price has recovered to 3505, which is > VWAP (3500)
    holdings = res_3["buy"] # Assuming the C++ hub executed our previous buy
    res_4 = strategy.execute(3505.0, vwap_px, pred_vol, regime_chop, capital, holdings)
    print(f"Result: {res_4}")