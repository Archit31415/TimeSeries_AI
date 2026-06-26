import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom architecture
from src.architecture import DualBranchVolatilityNet

def load_test_data(filepath):
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    return df

def run_backtest(df, model, device, initial_capital=10000.0, seq_len=78, sma_period=20):
    print(f"Starting backtest with ${initial_capital:,.2f} initial capital...")
    
    cash = initial_capital
    inventory = 0.0  
    equity_curve = []
    
    # New tracking metrics
    total_trades = 0
    total_fees_paid = 0.0
    
    # Standard exchange fee (Binance spot = 0.1%)
    fee_rate = 0.001 
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume_asset', 
        'sin_time', 'cos_time', 'variance_ratio', 
        'mfi', 'vol_acceleration'
    ]
    
    features_array = df[feature_cols].values
    close_prices = df['close'].values
    vr_values = df['variance_ratio'].values
    
    for i in range(seq_len, len(df)):
        current_price = close_prices[i]
        current_vr = vr_values[i]
        
        # 1. Model Prediction
        seq = features_array[i - seq_len : i]
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_rv, _ = model(x_tensor)
            
        predicted_vol = torch.sqrt(predicted_rv).item()
        
        # 2. Dynamic Bands
        sma_20 = np.mean(close_prices[i - sma_period : i])
        upper_band = sma_20 * (1 + (2 * predicted_vol))
        lower_band = sma_20 * (1 - (2 * predicted_vol))
        
        # 3. Trading Logic 
        if current_vr < 1.0: # Mean-Reverting
            if current_price <= lower_band:
                capital_to_risk = cash * 0.05
                qty = round(capital_to_risk / current_price, 4) 
                
                # Calculate fee exactly
                fee = (qty * current_price) * fee_rate
                cost = (qty * current_price) + fee
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    inventory += qty
                    total_fees_paid += fee
                    total_trades += 1
                    
            elif current_price >= upper_band and inventory > 0:
                fee = (inventory * current_price) * fee_rate
                revenue = (inventory * current_price) - fee
                
                cash += revenue
                total_fees_paid += fee
                total_trades += 1
                inventory = 0.0
                
        else: # Breakout / Trending
            if current_price >= upper_band:
                capital_to_risk = cash * 0.10
                qty = round(capital_to_risk / current_price, 4)
                
                fee = (qty * current_price) * fee_rate
                cost = (qty * current_price) + fee
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    inventory += qty
                    total_fees_paid += fee
                    total_trades += 1
                    
            elif current_price <= sma_20 and inventory > 0:
                fee = (inventory * current_price) * fee_rate
                revenue = (inventory * current_price) - fee
                
                cash += revenue
                total_fees_paid += fee
                total_trades += 1
                inventory = 0.0
                
        # 4. Log current equity
        current_equity = cash + (inventory * current_price)
        equity_curve.append(current_equity)
        
        if i % 5000 == 0:
            print(f"Step {i}/{len(df)} | Equity: ${current_equity:,.2f} | Hold: {inventory:.4f} ETH | Fees: ${total_fees_paid:.2f}")

    # Final liquidation
    if inventory > 0:
        final_price = close_prices[-1]
        fee = (inventory * final_price) * fee_rate
        cash += (inventory * final_price) - fee
        
        total_fees_paid += fee
        total_trades += 1
        inventory = 0.0
        equity_curve[-1] = cash # Update last point
        
    return pd.Series(equity_curve, index=df.index[seq_len:]), cash, total_trades, total_fees_paid

def calculate_metrics(equity_series, initial_cap):
    returns = equity_series.pct_change().dropna()
    bars_per_year = 365 * 24 * 12 
    
    total_return = ((equity_series.iloc[-1] - initial_cap) / initial_cap) * 100
    
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(bars_per_year) if std_ret > 0 else 0.0
    
    downside_returns = returns[returns < 0]
    down_std = downside_returns.std()
    sortino = (mean_ret / down_std) * np.sqrt(bars_per_year) if down_std > 0 else 0.0
    
    return total_return, max_drawdown, sharpe, sortino


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_path = "data/processed/test_processed.csv"
    weights_path = "models/dual_branch_weights.pth"
    
    seq_len = 78
    num_features = 10
    
    print("Loading Dual-Branch Volatility Network...")
    model = DualBranchVolatilityNet(num_features=num_features, seq_len=seq_len)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        print(f"Error: Weights not found at {weights_path}.")
        sys.exit(1)
        
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}.")
        sys.exit(1)
        
    df_test = load_test_data(test_data_path)
    
    initial_cap = 10000.0
    equity_series, final_balance, total_trades, total_fees = run_backtest(df_test, model, device, initial_capital=initial_cap)
    
    total_return, max_dd, sharpe, sortino = calculate_metrics(equity_series, initial_cap)
    
    print("\n" + "="*45)
    print("BACKTEST RESULTS (Jan 2026 - May 2026)")
    print("="*45)
    print(f"Initial Capital : ${initial_cap:,.2f}")
    print(f"Final Equity    : ${final_balance:,.2f}")
    print(f"Total Return    : {total_return:.2f}%")
    print(f"Max Drawdown    : {max_dd:.2f}%")
    print(f"Sharpe Ratio    : {sharpe:.2f}")
    print(f"Sortino Ratio   : {sortino:.2f}")
    print(f"Total Trades    : {total_trades}")
    print(f"Total Fees Paid : ${total_fees:,.2f}")
    print("="*45)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_series.index, equity_series.values, label='Portfolio Equity', color='blue')
    plt.title("Backtest Equity Curve: Dual-Branch Volatility Strategy")
    plt.ylabel("Account Balance (USD)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest_equity_curve.png")
    print("\nSaved equity curve plot to 'backtest_equity_curve.png'.")