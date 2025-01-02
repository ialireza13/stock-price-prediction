# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils.helpers import load_processed_data, load_model, make_forecast, get_available_tickers
from datetime import datetime, timedelta
import os
from data.fetch_data import fetch_and_save_stock_data
from data.preprocess_data import preprocess_stock_data
import joblib
from models.train_model import train_prophet_model  # Ensure this function is importable

# Function to ensure data is present
def ensure_data():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    for ticker in tickers:
        processed_file = os.path.join('data', 'processed', f"{ticker}_processed.csv")
        model_file = os.path.join('models', f"{ticker}_model.pkl")
        if not os.path.exists(processed_file):
            # Fetch raw data
            fetch_and_save_stock_data(ticker)
            # Preprocess data
            preprocess_stock_data(ticker)
            # Train model
            train_prophet_model(ticker)

# Ensure data is available
ensure_data()

st.set_page_config(page_title="Stock Price Prediction and Analysis", layout="wide")

st.title("📈 Stock Price Prediction and Analysis Tool")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    tickers = get_available_tickers()
    ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
    
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    
    prediction_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=60, value=30)
    
    return ticker, start_date, end_date, prediction_days

ticker, start_date, end_date, prediction_days = user_input_features()

# Load data
df = load_processed_data(ticker)
df['Date'] = pd.to_datetime(df['Date'])

# Filter data based on user input
mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
filtered_df = df.loc[mask]

# Plotting Historical Price
st.subheader(f"Historical Close Price for {ticker}")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], name="Close Price"))
fig1.update_layout(xaxis_rangeslider_visible=True)
st.plotly_chart(fig1, use_container_width=True)

# Plotting Moving Averages
st.subheader("Moving Averages")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA10'], name="MA 10"))
fig2.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA50'], name="MA 50"))
fig2.update_layout(xaxis_rangeslider_visible=True)
st.plotly_chart(fig2, use_container_width=True)

# Plotting RSI
st.subheader("Relative Strength Index (RSI)")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['RSI'], name="RSI"))
fig3.update_layout(xaxis_rangeslider_visible=True, yaxis=dict(range=[0, 100]))
st.plotly_chart(fig3, use_container_width=True)

# Prediction
st.subheader(f"Stock Price Prediction for next {prediction_days} days")

model = load_model(ticker)
forecast = make_forecast(model, periods=prediction_days)

# Plotting Forecast
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Close"))
fig4.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Close"))
fig4.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Confidence", line=dict(dash='dash')))
fig4.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Confidence", line=dict(dash='dash')))
fig4.update_layout(xaxis_rangeslider_visible=True)
st.plotly_chart(fig4, use_container_width=True)

# Display Forecast Data
st.subheader("Forecast Data")
st.write(forecast.tail(prediction_days))

# Technical Indicators
st.subheader("Technical Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Moving Average 10", f"{df['MA10'].iloc[-1]:.2f}")
with col2:
    st.metric("Moving Average 50", f"{df['MA50'].iloc[-1]:.2f}")
with col3:
    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

# Footer
st.markdown("""
---
Created by [Your Name](https://github.com/yourusername)
""")
