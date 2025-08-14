import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show critical errors

import time
from typing import Tuple
import bittensor as bt
import pandas as pd
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str
import csv
import numpy as np
import joblib
from tensorflow.keras.models import load_model

cached_price_data = None
last_fetch_time = 0
CACHE_DURATION_SECONDS = 60
model = None

def get_point_estimate(cm: CMData, timestamp: str) -> float:
    global cached_price_data, last_fetch_time

    provided_dt = to_datetime(timestamp)

    current_time = time.time()
    if cached_price_data is not None and (current_time - last_fetch_time) < CACHE_DURATION_SECONDS:
        bt.logging.info("Using cached price data.")
        price_data = cached_price_data
    else:
        bt.logging.info("Fetching new price data from API...")
        price_data = cm.get_CM_ReferenceRate(
            assets="BTC",
            start=None,
            end=to_str(provided_dt),
            frequency="1s",
            limit_per_asset=9600,
            paging_from="end",
            use_cache=False,
        )
        cached_price_data = price_data
        last_fetch_time = current_time

    SEQ_LENGTH = 150

    if isinstance(price_data, pd.DataFrame):
        series = price_data["ReferenceRateUSD"]
    else:
        series = price_data

    recent_series = series.iloc[-9300:]
    diff = recent_series.iloc[-300:].mean() - series.iloc[-3600:].mean()

    df = pd.DataFrame({
        'close': recent_series.values.reshape(-1, 60)[:,-1]
    }).dropna().reset_index(drop=True)

    if len(df) < SEQ_LENGTH:
        raise ValueError(f"Not enough data. Required: {SEQ_LENGTH}, Provided: {len(df)}")

    window = df.iloc[-SEQ_LENGTH:].copy()

    min_val = window['close'].min()
    max_val = window['close'].max()
    norm_window = (window - min_val) / (max_val - min_val + 1e-8)

    model_input = norm_window['close'].values.reshape(1, SEQ_LENGTH, 1).astype(np.float32)
    last_price = window['close'].iloc[-1]

    predict = model.predict(model_input)

    if predict is None:
        bt.logging.info(f"Model prediction failed.")
        return last_price, diff # Return a default value

    raw = np.argmax(predict, axis=1)[0]

    bt.logging.info(f" ========== Now Strategy ===========> {raw} <===========")

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

    num = np.random.uniform(-2, 2)
    return prediction + num, diff


def forward(synapse: Challenge, cm: CMData) -> Challenge:
    global model
    bt.logging.info(
        f"👈 {synapse.dendrite.hotkey} for {synapse.timestamp}"
    )

    if  model is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model = load_model(os.path.join(current_dir, "model/080414.h5"), compile=False)
        bt.logging.info("Keras model loaded successfully.")

    point_estimate, diff = get_point_estimate(cm=cm, timestamp=synapse.timestamp)

    interval_delta : float = 250
    if abs(diff) < 50 : interval_delta = 80
    elif abs(diff) < 100 : interval_delta = 170
    elif abs(diff) < 200 : interval_delta = 230
    elif abs(diff) < 350 : interval_delta = 300
    else: interval_delta = 380

    prediction_interval: Tuple[float, float] = (point_estimate - interval_delta, point_estimate + interval_delta)
    synapse.prediction = point_estimate
    synapse.interval = prediction_interval

    if synapse.prediction is not None:
        bt.logging.success(f"Predict => {synapse.prediction}  |  Interval => {synapse.interval} +- {interval_delta}")
        bt.logging.info("Classification model -1")

    else:
        bt.logging.info("No prediction for this request.")
    return synapse