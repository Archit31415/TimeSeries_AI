import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestMetrics:
    """
    Computes institutional-grade performance metrics for the trading strategy.
    Designed for 24/7 crypto markets (5-minute resolution).
    """
    def __init__(self, risk_free_rate: float = 0.0, fee_bps: float = 5.0):
        """
        Args:
            risk_free_rate: Annualized risk-free rate (default 0 for crypto, or use Treasury yield).
            fee_bps: Binance execution fee in basis points (e.g., 5.0 bps = 0.05%).
        """
        self.rf_rate = risk_free_rate
        self.fee_rate = fee_bps / 10000.0  # Convert bps to decimal
        
        # Crypto trades 24/7/365. 
        # 12 bars/hour * 24 hours * 365 days = 105,120 bars per year.
        self.annualization_factor = 105120  

    def calculate_sharpe_ratio(self, portfolio_returns: pd.Series) -> float:
        """
        Calculates the Annualized Sharpe Ratio.
        Formula: (Mean Return - Risk Free Rate) / Std Dev of Returns * sqrt(Annualization Factor)
        A Sharpe > 1.0 is acceptable, > 1.5 is excellent, > 2.0 is holy grail.
        """
        if portfolio_returns.std() == 0:
            return 0.0
            
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Adjust risk-free rate to the 5-min timeframe
        rf_per_period = self.rf_rate / self.annualization_factor
        
        sharpe = (mean_return - rf_per_period) / std_return
        annualized_sharpe = sharpe * np.sqrt(self.annualization_factor)
        
        return annualized_sharpe

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculates the Maximum Drawdown (MDD).
        The largest peak-to-trough percentage drop in portfolio value.
        """
        # Calculate the running maximum
        rolling_max = equity_curve.cummax()
        
        # Calculate the percentage drawdown from the rolling maximum
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        # MDD is the minimum value (most negative) in the drawdowns series
        max_dd = drawdowns.min()
        
        return abs(max_dd)

    def calculate_band_coverage(self, actual_returns: pd.Series, predicted_vols: pd.Series, z_score: float = 1.96) -> float:
        """
        The core evaluation metric for the Volatility LSTM.
        Checks what percentage of actual future returns landed inside the predicted dynamic bands.
        For a z_score of 1.96 (assuming normal distribution), target coverage is ~95%.
        """
        pred_5min_vol = predicted_vols / np.sqrt(288)
        
        # Upper and lower bounds predicted by the model
        upper_band = pred_5min_vol * z_score
        lower_band = -pred_5min_vol * z_score
        
        # Boolean mask of whether the return stayed inside the bands
        inside_bands = (actual_returns >= lower_band) & (actual_returns <= upper_band)
        
        coverage_pct = inside_bands.mean() * 100.0
        return coverage_pct

    def generate_trade_summary(self, trades: pd.DataFrame) -> dict:
        """
        Calculates execution-specific metrics (Win Rate, Profit Factor, Fees Paid).
        """
        # Define the default structure for when no trades occur
        default_stats = {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "total_fees_paid": 0.0
        }
        
        if trades.empty:
            return default_stats
            
        # Deduct fees from PnL
        trades['net_pnl'] = trades['pnl'] - (trades['volume_traded'] * self.fee_rate * 2)
        
        winning_trades = trades[trades['net_pnl'] > 0]
        losing_trades = trades[trades['net_pnl'] <= 0]
        
        win_rate = (len(winning_trades) / len(trades)) * 100.0
        
        gross_profit = winning_trades['net_pnl'].sum()
        gross_loss = abs(losing_trades['net_pnl'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        total_fees = (trades['volume_traded'] * self.fee_rate * 2).sum()
        
        return {
            "total_trades": len(trades),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "net_profit": trades['net_pnl'].sum(),
            "total_fees_paid": total_fees
        }

    def print_full_report(self, equity_curve: pd.Series, actual_returns: pd.Series, 
                          predicted_vols: pd.Series, trades: pd.DataFrame):
        """Prints a stylized console report for the backtest notebooks."""
        
        portfolio_returns = equity_curve.pct_change().dropna()
        
        sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        mdd = self.calculate_max_drawdown(equity_curve)
        coverage = self.calculate_band_coverage(actual_returns, predicted_vols)
        trade_stats = self.generate_trade_summary(trades)
        
        print("\n" + "="*50)
        print(" DEEPVOL BACKTEST PERFORMANCE REPORT")
        print("="*50)
        print(f" Initial Capital:    ${equity_curve.iloc[0]:,.2f}")
        print(f" Final Capital:      ${equity_curve.iloc[-1]:,.2f}")
        print(f" Net Return:         {((equity_curve.iloc[-1]/equity_curve.iloc[0])-1)*100:.2f}%")
        print("-" * 50)
        print(f" Annualized Sharpe:  {sharpe:.3f}")
        print(f" Maximum Drawdown:   {mdd*100:.2f}%")
        print(f" Band Coverage:      {coverage:.2f}% (Target: ~95.0%)")
        print("-" * 50)
        print(f" Total Trades:       {trade_stats['total_trades']}")
        print(f" Win Rate:           {trade_stats['win_rate_pct']:.1f}%")
        print(f" Profit Factor:      {trade_stats['profit_factor']:.2f}")
        print(f" Est. Exchange Fees: ${trade_stats['total_fees_paid']:,.2f}")
        print("="*50 + "\n")

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate a backtest result for testing
    dates = pd.date_range("2025-01-01", periods=1000, freq="5min")
    
    # 1. Simulate Equity Curve (Starting at $10k, drifting upward with some noise)
    random_returns = np.random.normal(0.00001, 0.001, 1000)
    equity = pd.Series(10000 * np.cumprod(1 + random_returns), index=dates)
    
    # 2. Simulate Volatility Band Coverage check
    actual_ret = pd.Series(np.random.normal(0, 0.005, 1000), index=dates)
    pred_vol = pd.Series(np.full(1000, 0.0055), index=dates) # Model predicted slightly higher vol
    
    # 3. Simulate Trade Log
    trades_df = pd.DataFrame({
        'pnl': np.random.normal(10, 50, 50),        # 50 trades, avg $10 profit, high variance
        'volume_traded': np.random.uniform(1000, 5000, 50) # Dollar volume per trade
    })
    
    # Initialize grading module with Binance Futures Maker/Taker blended rate (~4 bps)
    grader = BacktestMetrics(fee_bps=4.0)
    
    # Generate Report
    grader.print_full_report(equity, actual_ret, pred_vol, trades_df)