import requests
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Binance REST API Endpoint
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance_klines(symbol: str, interval: str, start_ts: int, end_ts: int, limit: int = 1000):
    """
    Fetch a single batch of klines from Binance.
    Binance limit is 1000 bars per request.
    """
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': limit
    }
    
    try:
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API Error fetching {symbol}: {e}")
        time.sleep(5)  # Back off on rate limits or connection errors
        return None

def download_historical_data(symbol: str, start_date: str, end_date: str, interval: str = '5m', save_dir: str = '../../data/raw'):
    """
    Orchestrates the downloading of historical data by paginating through time.
    """
    # Convert string dates to milliseconds timestamps (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    current_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    all_klines = []
    
    logging.info(f"Starting fetch for {symbol} {interval} from {start_date} to {end_date}...")
    
    while current_ts < end_ts:
        klines = fetch_binance_klines(symbol, interval, current_ts, end_ts)
        
        if not klines:
            # If nothing returned, push timestamp forward slightly to avoid infinite loops on dead periods
            current_ts += 60000 * 5 
            continue
            
        all_klines.extend(klines)
        
        # Update current_ts to the last fetched candle's close time + 1 millisecond
        last_candle_close_time = klines[-1][6]
        current_ts = last_candle_close_time + 1
        
        # Be polite to the public API
        time.sleep(0.1)
        
        if len(all_klines) % 50000 == 0:
            logging.info(f"Fetched {len(all_klines)} bars so far. Current date: {datetime.fromtimestamp(current_ts/1000, tz=timezone.utc)}")

    # Binance Kline structure:
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume_asset',
        'close_time', 'volume_usdt', 'trade_count', 
        'taker_buy_asset', 'taker_buy_usdt', 'ignore'
    ]
    
    df = pd.DataFrame(all_klines, columns=columns)
    
    # Clean up and typecast the DataFrame
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=['close_time', 'ignore'], inplace=True)
    
    # Convert string metrics to float/int
    float_cols = ['open', 'high', 'low', 'close', 'volume_asset', 'volume_usdt', 'taker_buy_asset', 'taker_buy_usdt']
    df[float_cols] = df[float_cols].astype(float)
    df['trade_count'] = df['trade_count'].astype(int)
    
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    filename = f"{symbol}_{interval}_{start_date}_to_{end_date}.csv"
    filepath = Path(save_dir) / filename
    df.to_csv(filepath)
    
    logging.info(f"Successfully saved {len(df)} bars to {filepath}")
    return df

if __name__ == "__main__":
    # Ensure we are saving relative to the script location assuming it's in src/data_pipeline/
    target_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw'
    
    # --- PHASE 1: Training & Validation Set (Pre-2026) ---
    # Used for fitting scalers, HMM regimes, and training LSTMs
    download_historical_data(
        symbol="ETHUSDT",
        start_date="2024-06-01",
        end_date="2025-12-31",
        interval="5m",
        save_dir=str(target_dir)
    )
    
    # --- PHASE 2: Strict OOS Test Set (2026) ---
    # Locked away until the final backtest in Guwahati
    download_historical_data(
        symbol="ETHUSDT",
        start_date="2026-01-01",
        end_date="2026-06-01",
        interval="5m",
        save_dir=str(target_dir)
    )
    
    # Optional: Fetch BTC for multi-asset correlation (if Divyam needs it for Idea 1)
    # download_historical_data("BTCUSDT", "2024-06-01", "2025-12-31", "5m", str(target_dir))