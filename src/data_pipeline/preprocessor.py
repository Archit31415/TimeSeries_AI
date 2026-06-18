import pandas as pd
import numpy as np
import pywt
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketDataPreprocessor:
    def __init__(self):
        """
        Initializes the preprocessor. 
        The StandardScaler is instantiated here but MUST only be fitted on training data.
        """
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def force_time_grid(self, df: pd.DataFrame, interval: str = '5min') -> pd.DataFrame:
        """
        Forces a perfectly continuous time grid. Crypto APIs sometimes skip bars 
        if 0 trades occurred. We must fill these to maintain the LSTM's temporal spacing.
        """
        logging.info("Forcing continuous time grid...")
        
        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep='last')].sort_index()
        
        # Create a perfect theoretical grid from start to finish
        full_grid = pd.date_range(start=df.index.min(), end=df.index.max(), freq=interval)
        
        # Reindex to the perfect grid
        df = df.reindex(full_grid)
        
        # 1. Forward-fill prices (if no trades happened, the price stays the same)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # 2. Fill volume and trade counts with 0 (if no trades happened, volume is 0)
        vol_cols = ['volume_asset', 'volume_usdt', 'trade_count', 'taker_buy_asset', 'taker_buy_usdt']
        df[vol_cols] = df[vol_cols].fillna(0)
        
        # Drop any remaining NaNs at the very beginning (if first row was NaN)
        df.dropna(inplace=True)
        logging.info(f"Grid fill complete. Final shape: {df.shape}")
        
        return df

    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates logarithmic returns. Neural networks require stationary data.
        Raw prices will cause exploding gradients and structural bias.
        """
        logging.info("Computing Log Returns...")
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Log return of volume (adding 1e-8 to avoid log(0) on empty grid bars)
        df['log_volume_usdt'] = np.log(df['volume_usdt'] + 1e-8)
        
        df.dropna(inplace=True) # Drop the first row which will have NaN return
        return df

    def rolling_wavelet_denoise(self, series: pd.Series, window: int = 78, wavelet: str = 'db4') -> pd.Series:
        """
        Applies Discrete Wavelet Transform (DWT) to denoise a signal.
        CRITICAL: Standard DWT uses the whole array (future data) to smooth the past. 
        To prevent lookahead bias, we MUST apply this on a rolling window, 
        only looking at the last `window` bars (78 bars = 1 trading day roughly).
        """
        logging.info(f"Applying rolling DWT denoising (window={window}, wavelet={wavelet}). This may take a moment...")
        
        denoised_signal = np.full(len(series), np.nan)
        values = series.values
        
        for i in range(window, len(values)):
            # Extract strictly historical window
            current_window = values[i-window:i]
            
            # Decompose signal
            coeffs = pywt.wavedec(current_window, wavelet, level=2)
            
            # Thresholding: Zero out the high-frequency detail coefficients (noise)
            # coeffs[0] is the approximation (trend), the rest are details (noise)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust MAD estimator
            uthresh = sigma * np.sqrt(2 * np.log(len(current_window)))
            
            # Apply soft thresholding to detail coefficients
            coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
            
            # Reconstruct the signal
            reconstructed = pywt.waverec(coeffs, wavelet)
            
            # Take ONLY the final point of the reconstructed window (t) to prevent leakage
            denoised_signal[i] = reconstructed[-1]
            
        return pd.Series(denoised_signal, index=series.index)

    def fit_transform_scaler(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Fits the scaler on the TRAINING data and transforms it.
        """
        logging.info("Fitting and applying StandardScaler (Training Mode)...")
        df_scaled = df.copy()
        
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        self.is_fitted = True
        
        return df_scaled

    def transform_scaler(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Applies the previously fitted scaler to VALIDATION or TEST data.
        Raises an error if the scaler hasn't been fitted yet to protect the pipeline.
        """
        if not self.is_fitted:
            raise ValueError("CRITICAL: Scaler has not been fitted! Run fit_transform_scaler on train set first.")
            
        logging.info("Applying StandardScaler (Inference/Test Mode)...")
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df_scaled

# --- Example Usage / Sanity Check ---
if __name__ == "__main__":
    # Simulate loading the raw data from fetch_binance.py
    # df_raw_train = pd.read_csv('../../data/raw/ETHUSDT_5m_2024-06-01_to_2025-12-31.csv', index_col=0, parse_dates=True)
    
    # Example dummy execution:
    processor = MarketDataPreprocessor()
    
    print("Preprocessor initialized successfully. Ready to process raw datasets.")
    print("Methods available:")
    print("1. processor.force_time_grid(df)")
    print("2. processor.compute_log_returns(df)")
    print("3. processor.rolling_wavelet_denoise(df['log_return'])")
    print("4. processor.fit_transform_scaler(train_df, feature_cols)")
    print("5. processor.transform_scaler(test_df, feature_cols)")