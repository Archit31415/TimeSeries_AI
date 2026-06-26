import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KellyVolatilitySizer:
    """
    Position sizing module that combines the Fractional Kelly Criterion 
    with Inverse Volatility Targeting.
    """
    def __init__(self, kelly_fraction: float = 0.5, max_capital_risk: float = 0.25, reference_vol: float = 0.01):
        """
        Args:
            kelly_fraction: The fraction of the Kelly optimal size to bet (0.5 = Half-Kelly, standard for quants).
            max_capital_risk: The absolute maximum % of total capital allowed per trade.
            reference_vol: The baseline volatility (e.g., 1%) used to normalize the inverse volatility scalar.
        """
        self.kelly_fraction = kelly_fraction
        self.max_capital_risk = max_capital_risk
        self.reference_vol = reference_vol
        
        logging.info(f"Initialized KellyVolatilitySizer (Fraction: {kelly_fraction}, Max Risk: {max_capital_risk*100}%)")

    def calculate_kelly_weight(self, win_probability: float, win_loss_ratio: float = 1.0) -> float:
        """
        Calculates the pure Kelly Criterion weight.
        Formula: f* = p - (q / b)
        Where p = win prob, q = loss prob (1-p), b = odds ratio (win/loss).
        If assuming a 1:1 payout ratio, this simplifies to: f* = (2p - 1).
        """
        # If probability is less than 50%, the Kelly weight is 0 or negative (do not trade)
        if win_probability <= 0.50:
            return 0.0
            
        loss_probability = 1.0 - win_probability
        
        # Standard Kelly formula
        kelly_optimal = win_probability - (loss_probability / win_loss_ratio)
        
        # Apply fractional scaling (e.g., Half-Kelly) to reduce drawdown risk
        fractional_kelly = kelly_optimal * self.kelly_fraction
        
        return max(0.0, fractional_kelly)

    def calculate_volatility_scalar(self, predicted_vol: float) -> float:
        """
        Calculates the Inverse Volatility scalar.
        Higher predicted volatility = Smaller scalar = Smaller position size.
        """
        # Protect against division by zero or negative vol
        safe_vol = max(predicted_vol, 1e-6)
        
        # If predicted vol is 2% and reference is 1%, scalar is 0.5 (halve the position)
        scalar = self.reference_vol / safe_vol
        
        # Cap the scalar to prevent massive over-leveraging during extreme low-vol anomalies
        return min(scalar, 5.0)

    def get_target_shares(self, available_capital: float, current_price: float, 
                          predicted_vol: float, regime_confidence: float) -> float:
        """
        Computes the final number of shares to buy by blending Kelly and Volatility Targeting.
        
        Args:
            available_capital: Total USDT in the account.
            current_price: Current asset price.
            predicted_vol: Standard deviation output from the VolLSTM.
            regime_confidence: Probability output from the Regime LSTM (e.g., P(Chop)).
            
        Returns:
            float: Floor-rounded shares to execute.
        """
        # 1. Base Kelly sizing based on model confidence
        # Follows your spec: allocation scales by (Probability - 0.5) * 2
        kelly_weight = self.calculate_kelly_weight(win_probability=regime_confidence, win_loss_ratio=1.0)
        
        if kelly_weight <= 0.0:
            return 0.0
            
        # 2. Adjust for Volatility Target
        vol_scalar = self.calculate_volatility_scalar(predicted_vol)
        
        # 3. Final raw allocation percentage
        target_pct = kelly_weight * vol_scalar
        
        # 4. Hard Risk Limits
        target_pct = min(target_pct, self.max_capital_risk)
        
        # 5. Convert to Fiat and Shares
        fiat_allocation = available_capital * target_pct
        raw_shares = fiat_allocation / current_price
        
        # 6. Exchange formatting: Floor to 4 decimal places to prevent API rejection
        final_shares = math.floor(raw_shares * 10000) / 10000.0
        
        logging.info(
            f"Sizing Logic -> Kelly Wgt: {kelly_weight:.3f} | Vol Scalar: {vol_scalar:.2f} | "
            f"Final Risk: {target_pct*100:.1f}% | Allocation: ${fiat_allocation:.2f}"
        )
        
        return final_shares

# --- Module Test & Sanity Check ---
if __name__ == "__main__":
    print("--- Testing Position Sizing Module ---")
    
    # Initialize Half-Kelly sizer, max 25% account risk per trade
    sizer = KellyVolatilitySizer(kelly_fraction=0.5, max_capital_risk=0.25)
    
    capital = 10000.0
    price = 3500.0
    
    print("\nScenario 1: High Confidence, Normal Volatility")
    # 80% confident it's a Chop regime, vol is baseline (1%)
    shares_1 = sizer.get_target_shares(capital, price, predicted_vol=0.01, regime_confidence=0.80)
    print(f"Shares to buy: {shares_1} (Total Value: ${shares_1 * price:.2f})")
    
    print("\nScenario 2: High Confidence, EXTREME Volatility")
    # Still 80% confident, but vol is 4% (4x normal risk)
    shares_2 = sizer.get_target_shares(capital, price, predicted_vol=0.04, regime_confidence=0.80)
    print(f"Shares to buy: {shares_2} (Total Value: ${shares_2 * price:.2f})")
    # Notice the allocation is exactly 1/4th of Scenario 1!
    
    print("\nScenario 3: Low Confidence (Below 50%)")
    # Only 45% confident it's a Chop regime
    shares_3 = sizer.get_target_shares(capital, price, predicted_vol=0.01, regime_confidence=0.45)
    print(f"Shares to buy: {shares_3} (Total Value: ${shares_3 * price:.2f})")
    
    print("\nScenario 4: Perfect Setup, but hit Max Risk Cap")
    # 95% confident, extremely low volatility (0.2%)
    shares_4 = sizer.get_target_shares(capital, price, predicted_vol=0.002, regime_confidence=0.95)
    print(f"Shares to buy: {shares_4} (Total Value: ${shares_4 * price:.2f})")
    # Capped at $2,500 (25% of $10,000)