import os
import pandas as pd
from prophet import Prophet
import joblib

PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_DIR = 'models'

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_processed_data(ticker):
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data for {ticker} not found.")
    df = pd.read_csv(processed_file)
    # Ensure 'Date' is datetime and timezone-naive
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    return df

def load_model(ticker):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker} not found.")
    return joblib.load(model_path)

def make_forecast(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def get_available_tickers():
    processed_files = os.listdir(PROCESSED_DATA_DIR)
    tickers = [file.split('_')[0] for file in processed_files if file.endswith('_processed.csv')]
    return tickers
