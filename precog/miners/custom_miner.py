import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show critical errors

import time
from typing import Tuple, Optional
import bittensor as bt
import pandas as pd
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str
import numpy as np
from tensorflow.keras.models import load_model
import requests
from binance.client import Client

# --- Globals ---
cached_price_data: Optional[pd.Series] = None
last_fetch_time: float = 0
CACHE_DURATION_SECONDS: int = 60
model = None
binance_client = Client()

# --- New Helper Function to get live USDT to USD rate ---
def get_usdt_to_usd_rate() -> float:
    """
    Fetches the live USDT to USD conversion rate from Kraken's public API.
    Returns 1.0 as a fallback if the API call fails.
    """
    try:
        url = "https://api.kraken.com/0/public/Ticker?pair=USDTUSD"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        price_str = data['result']['USDTZUSD']['c'][0]
        return float(price_str)
    except Exception as e:
        bt.logging.warning(f"Could not fetch live USDT/USD rate, defaulting to 1.0. Error: {e}")
        return 1.0

# --- New Main data fetching function ---
def get_price_series_from_api(end_timestamp: str) -> Optional[pd.Series]:
    """
    Fetches 1-minute BTC/USDT price data from Binance and uses a live
    rate to convert it to an approximate BTC/USD price.
    """
    try:
        usdt_to_usd_rate = get_usdt_to_usd_rate()
        bt.logging.info(f"Using dynamic USDT -> USD rate: {usdt_to_usd_rate}")

        end_dt = to_datetime(end_timestamp)
        end_ms = int(end_dt.timestamp() * 1000)

        klines = binance_client.get_klines(
            symbol='BTCUSDT',
            interval=Client.KLINE_INTERVAL_1MINUTE,
            limit=300,  # Fetch 300 minutes to be safe
            endTime=end_ms
        )
        
        if not klines:
            bt.logging.warning("No data returned from Binance API.")
            return None

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        price_series = pd.to_numeric(df['close']) * usdt_to_usd_rate
        price_series.name = "ReferenceRateUSD"
        return price_series

    except Exception as e:
        bt.logging.error(f"Error fetching data from API: {e}")
        return None

# --- Modified Prediction Function ---
def get_point_estimate(cm: CMData, timestamp: str) -> Tuple[float, float, float]:
    """
    This function is now powered by the Binance API with live conversion.
    The logic has been adapted from 1-second indexing to 1-minute indexing.
    The 'cm' parameter is unused but kept for signature compatibility.
    """
    global cached_price_data, last_fetch_time
    
    current_time = time.time()
    if cached_price_data is not None and (current_time - last_fetch_time) < CACHE_DURATION_SECONDS:
        bt.logging.info("Using cached price data.")
        series = cached_price_data
    else:
        bt.logging.info("Fetching new price data from API...")
        series = get_price_series_from_api(end_timestamp=timestamp)
        if series is None:
            raise ValueError("Failed to fetch price data from API.")
        cached_price_data = series
        last_fetch_time = current_time

    # The old code reshaped 1s data to 1m. We now have 1m data directly.
    df = pd.DataFrame({'close': series.values}).dropna().reset_index(drop=True)

    SEQ_LENGTH = 150
    if len(df) < SEQ_LENGTH:
        raise ValueError(f"Not enough data. Required: {SEQ_LENGTH}, Provided: {len(df)}")

    window = df.iloc[-SEQ_LENGTH:].copy()

    min_val, max_val = window['close'].min(), window['close'].max()
    norm_window = (window - min_val) / (max_val - min_val + 1e-8)
    
    model_input = norm_window['close'].values.reshape(1, SEQ_LENGTH, 1).astype(np.float32)
    last_price = window['close'].iloc[-1]
    
    predict = model.predict(model_input)

    if predict is None:
        bt.logging.info(f"Model prediction failed.")
        return last_price, last_price, last_price # Return a default tuple

    raw = np.argmax(predict, axis=1)[0]
    bt.logging.info(f" ========== Now Strategy ===========> {raw} <===========")
    
    # Your prediction logic remains unchanged
    if raw == 0:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-60:-5].mean())
    elif raw == 1:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-60:-5].mean()) * 0.7
    elif raw == 2:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-37:-5].mean())
    elif raw == 3:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-37:-5].mean()) * 0.35
    elif raw == 4:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-31:-5].mean())
    elif raw == 5:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-31:-5].mean()) * 0.38
    elif raw == 6:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-20:-5].mean())
    elif raw == 7:
        prediction = last_price + (window['close'].iloc[-5:].mean() - window['close'].iloc[-20:-5].mean()) * 0.6
    else:
        prediction = last_price
    
    # Adapted for 1-minute data: 3600 seconds -> last 60 minutes
    std: float = series.iloc[-60:].std()
    mean: float = series.iloc[-60:].mean()

    delta: float = prediction - last_price
    interval_top: float = mean + delta + 2 * std
    interval_bottom: float = mean + delta - 2 * std

    return prediction + 0.31, interval_top + 0.31, interval_bottom + 0.31

# --- Unchanged Main Function ---
def forward(synapse: Challenge, cm: CMData) -> Challenge:
    global model
    bt.logging.info(
        f"👈 {synapse.dendrite.hotkey} for {synapse.timestamp}"
    )
    
    if model is None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "model/080414.h5")
        except NameError:
            model_path = "model/080414.h5"
            
        model = load_model(model_path, compile=False)
        bt.logging.info("Keras model loaded successfully.")

    try:
        point_estimate, interval_top, interval_bottom = get_point_estimate(cm=cm, timestamp=synapse.timestamp)
        
        prediction_interval: Tuple[float, float] = (interval_bottom, interval_top)
        synapse.prediction = point_estimate
        synapse.interval = prediction_interval

        if synapse.prediction is not None:
            bt.logging.success(f"Predict => {synapse.prediction}  |  Interval => {synapse.interval} ")
            bt.logging.info(f"Classification model with updated interval logic")
        else:
             bt.logging.info("No prediction for this request.")
             
    except ValueError as e:
        bt.logging.error(f"Prediction failed: {e}")
        synapse.prediction = None
        synapse.interval = (0.0, 0.0)

    return synapse