import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Expanded list of stock tickers (updated 'FB' to 'META')
STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'META', 'NFLX', 'NVDA', 'JPM', 'V', 
    'DIS', 'ADBE', 'PYPL', 'INTC', 'CSCO',
    'CMCSA', 'PEP', 'COST', 'AMGN', 'T'
]

# Define the date range
START_DATE = '2010-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Directory to save raw data
RAW_DATA_DIR = os.path.join('data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_and_save_stock_data(ticker):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=START_DATE, end=END_DATE)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {ticker} data to {file_path}")

def main():
    for ticker in STOCKS:
        fetch_and_save_stock_data(ticker)

if __name__ == "__main__":
    main()
