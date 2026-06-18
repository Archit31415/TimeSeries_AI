import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MicrostructureFeatures:
    """
    Computes market microstructure features that quantify institutional flow
    and price impact. Designed to identify 'smart money' vs. 'retail noise'.
    """

    def compute_whale_tracker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Average Trade Size and Z-Scores for volume.
        High average trade size with low tick count indicates institutional block orders.
        """
        logging.info("Computing Whale Tracker features...")
        
        # 1. Average Trade Size: Total Volume / Number of Transactions
        # Adding 1e-8 to prevent division by zero on low-liquidity bars
        df['avg_trade_size'] = df['volume_usdt'] / (df['trade_count'] + 1e-8)
        
        # 2. Volume Z-Score (Rolling 20-bar baseline)
        # Standardizes activity spikes independent of historical regime
        vol_mean = df['volume_usdt'].rolling(window=20).mean()
        vol_std = df['volume_usdt'].rolling(window=20).std()
        df['volume_z_score'] = (df['volume_usdt'] - vol_mean) / (vol_std + 1e-8)
        
        return df

    def compute_illiquidity_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Amihud Illiquidity: Price impact per dollar invested.
        High Amihud values indicate thin order books (fragile markets).
        """
        logging.info("Computing Amihud Illiquidity...")
        
        # Amihud = abs(Return) / Volume
        # Quantifies if a price move was backed by genuine liquidity or just a thin book
        df['amihud_illiquidity'] = np.abs(df['log_return']) / (df['volume_usdt'] + 1e-8)
        
        return df

    def compute_volume_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measures the spatial relationship between the candle body and wick.
        Bounded between -1 (Seller dominance) and +1 (Buyer dominance).
        """
        logging.info("Computing Volume Imbalance...")
        
        # (Close - Open) / (High - Low) represents directional pressure inside the candle
        range_val = df['high'] - df['low']
        df['vol_imbalance'] = (df['close'] - df['open']) / (range_val + 1e-8)
        
        return df

    def compute_vol_to_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measures market efficiency: Volatility generated per unit of capital spent.
        Declining volume with high volatility = Fragile market state.
        """
        logging.info("Computing Volatility-to-Volume Ratio...")
        
        # Uses Garman-Klass Volatility (assuming it was added in technicals.py)
        if 'garman_klass_vol' not in df.columns:
            logging.warning("Garman-Klass Volatility not found. Skipping ratio calculation.")
            return df
            
        df['vol_to_vol_ratio'] = df['garman_klass_vol'] / (df['volume_usdt'] + 1e-8)
        
        return df

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master method to execute the microstructure feature pipeline.
        """
        df = df.copy()
        
        df = self.compute_whale_tracker(df)
        df = self.compute_illiquidity_proxy(df)
        df = self.compute_volume_imbalance(df)
        df = self.compute_vol_to_volume_ratio(df)
        
        return df

if __name__ == "__main__":
    print("Microstructure feature module initialized.")