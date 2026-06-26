import pandas as pd
import numpy as np
from arch import arch_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_garch_baseline(train_series: pd.Series):
    """
    Trains a GARCH(1,1) model. 
    GARCH(1,1) is the standard statistical baseline for realized volatility.
    
    Args:
        train_series: Log returns of the asset.
    Returns:
        model_fit: The fitted GARCH model object.
    """
    logging.info("Training GARCH(1,1) baseline model...")
    
    # GARCH(1,1) assumes mean zero and constant volatility per period
    # 'vol="GARCH"' and 'p=1, q=1' is the classic configuration
    model = arch_model(train_series, vol='GARCH', p=1, q=1, dist='Normal')
    model_fit = model.fit(disp='off')
    
    logging.info(f"GARCH model converged. Summary:\n{model_fit.summary()}")
    return model_fit

def forecast_garch(model_fit, horizon: int = 1):
    """
    Produces one-step-ahead volatility forecast.
    """
    forecasts = model_fit.forecast(horizon=horizon)
    # The variance is in the 'variance' column; we take the square root for volatility
    cond_vol = np.sqrt(forecasts.variance.iloc[-1, 0])
    return cond_vol

if __name__ == "__main__":
    # Example usage logic
    # 1. Load your processed training data
    # df = pd.read_csv('data/processed/eth_usdt_train.csv', index_col=0, parse_dates=True)
    
    # 2. Train on log returns
    # garch_fit = train_garch_baseline(df['log_return'])
    
    # 3. Forecast volatility
    # vol_pred = forecast_garch(garch_fit)
    # print(f"GARCH Predicted Volatility: {vol_pred}")
    
    print("Baseline GARCH module initialized. Install via: pip install arch")