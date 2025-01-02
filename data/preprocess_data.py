import os
import pandas as pd

RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_stock_data(ticker):
    raw_file = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    processed_file = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    
    df = pd.read_csv(raw_file)
    
    # Convert 'Date' to datetime and ensure it's timezone-naive
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    
    # Feature Engineering: Add Moving Averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Add RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    
    # Drop NaN values resulting from rolling calculations
    df.dropna(inplace=True)
    
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

def main():
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith('.csv'):
            ticker = file.split('.')[0]
            preprocess_stock_data(ticker)

if __name__ == "__main__":
    main()
