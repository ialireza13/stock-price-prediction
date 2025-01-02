import os
import pandas as pd
from prophet import Prophet
import joblib
from datetime import datetime, timedelta

MODEL_DIR = 'models'

def predict_future(ticker, periods=30):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        print(f"Model for {ticker} does not exist. Please train the model first.")
        return None
    
    model = joblib.load(model_path)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def main():
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    periods = int(input("Enter number of days to predict (e.g., 30): "))
    forecast = predict_future(ticker, periods)
    if forecast is not None:
        print(forecast)

if __name__ == "__main__":
    main()
