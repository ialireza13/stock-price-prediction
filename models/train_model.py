import os
import pandas as pd
from prophet import Prophet
import joblib

PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_prophet_model(ticker):
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    df = pd.read_csv(processed_file)
    
    # Convert 'Date' to datetime and ensure it's timezone-naive
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    
    # Verify that 'Date' is timezone-naive
    if df['Date'].dt.tz is not None:
        raise ValueError(f"'Date' column for {ticker} is still timezone-aware.")
    
    # Prepare data for Prophet
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Debug: Check the first few rows
    print(f"Training Prophet model for {ticker}")
    
    model = Prophet()
    model.fit(prophet_df)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")

def main():
    for file in os.listdir(PROCESSED_DATA_DIR):
        if file.endswith('_processed.csv'):
            ticker = file.split('_')[0]
            train_prophet_model(ticker)

if __name__ == "__main__":
    main()