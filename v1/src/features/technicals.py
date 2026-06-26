import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TechnicalFeatures:
    """
    Computes technical and price-shape features for a 5-minute time grid.
    Expects a DataFrame that has already passed through `preprocessor.py` 
    (meaning it has a continuous time index and a 'log_return' column).
    """
    
    def __init__(self, bars_per_day=288):
        # 24 hours * 12 (5-min bars per hour) = 288 bars per day
        self.bars_per_day = bars_per_day

    def compute_har_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the classic Heterogeneous Autoregressive (HAR) realized volatility features.
        These capture the short (1 hour), medium (1 day), and long-term (1 week) memory 
        of volatility clustering.
        """
        logging.info("Computing HAR Volatility lags (1h, 1d, 1w)...")
        
        if 'log_return' not in df.columns:
            raise KeyError("Column 'log_return' is missing. Run preprocessor first.")
            
        # Squared returns are the basis of Realized Variance
        r2 = df['log_return'] ** 2
        
        # 1 Hour = 12 bars, 1 Day = 288 bars, 1 Week = 2016 bars
        df['rv_1h'] = np.sqrt(r2.rolling(window=12).sum())
        df['rv_1d'] = np.sqrt(r2.rolling(window=self.bars_per_day).sum())
        df['rv_1w'] = np.sqrt(r2.rolling(window=self.bars_per_day * 7).sum())
        
        return df

    def compute_garman_klass(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Garman-Klass Volatility.
        Unlike close-to-close returns, GK incorporates Open, High, Low, and Close 
        to capture hidden intra-candle velocity and chaos.
        """
        logging.info("Computing Garman-Klass Volatility...")
        
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        
        df['garman_klass_vol'] = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        return df

    def compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates scale-free momentum indicators (RSI and EMA ratio).
        We use the EMA ratio instead of raw EMA so the neural network 
        evaluates price structure independent of absolute price levels.
        """
        logging.info("Computing RSI and EMA Ratio...")
        
        # --- RSI (14 period) ---
        n = 14
        delta = df['close'].diff()
        
        # clip(lower=0) isolates gains, clip(upper=0) isolates losses
        up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        
        rs = up / dn.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # --- EMA Ratio (50 period) ---
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_ratio'] = (df['close'] / ema_50) - 1.0  # Centers around 0
        
        return df

    def compute_time_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes time of day and day of week as continuous cyclical features using sin/cos.
        This prevents the LSTM from seeing a hard mathematical break between 23:55 and 00:00.
        """
        logging.info("Computing Sin/Cos seasonality encodings...")
        
        # Minutes elapsed since midnight
        mins = df.index.hour * 60 + df.index.minute
        
        # Daily cycle
        df['sin_day'] = np.sin(2 * np.pi * mins / 1440)
        df['cos_day'] = np.cos(2 * np.pi * mins / 1440)
        
        # Weekly cycle (dow is 0-6, plus fractional day progress)
        dow = df.index.dayofweek + (mins / 1440)
        df['sin_week'] = np.sin(2 * np.pi * dow / 7)
        df['cos_week'] = np.cos(2 * np.pi * dow / 7)
        
        return df

    def generate_all(self, df: pd.DataFrame, drop_nans: bool = True) -> pd.DataFrame:
        """
        Master method to execute the full technical feature pipeline.
        """
        df = df.copy()
        
        df = self.compute_har_volatility(df)
        df = self.compute_garman_klass(df)
        df = self.compute_momentum_indicators(df)
        df = self.compute_time_encodings(df)
        
        if drop_nans:
            # Dropping NaNs here removes the initial warm-up period 
            # (e.g., the first 1 week of data needed to compute rv_1w)
            initial_len = len(df)
            df.dropna(inplace=True)
            dropped = initial_len - len(df)
            logging.info(f"Dropped {dropped} rows due to feature warm-up periods.")
            
        return df

# --- Example Usage ---
if __name__ == "__main__":
    # Assuming `df_processed` is output from preprocessor.force_time_grid() and compute_log_returns()
    # feature_builder = TechnicalFeatures(bars_per_day=288)
    # df_tech = feature_builder.generate_all(df_processed)
    # print(df_tech[['rv_1w', 'garman_klass_vol', 'rsi_14', 'sin_day']].head())
    print("Technical features module initialized.")