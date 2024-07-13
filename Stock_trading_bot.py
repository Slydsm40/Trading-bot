import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
import alpaca_trade_api as tradeapi
import logging
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from twilio.rest import Client  # Import Twilio client

# Twilio credentials
TWILIO_ACCOUNT_SID = 'your_twilio_account_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
RECIPIENT_PHONE_NUMBER = 'your_recipient_phone_number'  # Format: '+1234567890'

# Set up Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Set up logging
logging.basicConfig(filename='trading.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Alpaca API credentials (replace with your own)
ALPACA_API_KEY = 'YOUR_ALPACA_API_KEY'
ALPACA_SECRET_KEY = 'YOUR_ALPACA_SECRET_KEY'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# Function to fetch stock data
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = dropna(data)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train LSTM model
def train_lstm_model(X_train, y_train):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=1, epochs=10)
    return model

# Function to generate buy signal based on LSTM prediction
def generate_buy_signal(model, data, scaler):
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, len(data.columns)))
    X_test = np.array([last_60_days_scaled])
    predicted_price = model.predict(X_test)
    return predicted_price[0][0]

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values[-1]

# Function to check breakout resistance
def check_breakout_resistance(data):
    # Define breakout condition (e.g., close price above recent high)
    recent_high = data['High'].rolling(window=20).max().iloc[-1]
    current_close = data['Close'].iloc[-1]
    
    if current_close > recent_high:
        return True
    else:
        return False

# Function to send SMS notification
def send_sms_notification(message):
    try:
        client.messages.create(
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            body=message
        )
        logging.info(f'SMS notification sent: {message}')
    except Exception as e:
        logging.error(f'Error sending SMS notification: {str(e)}')

# Function to fetch sentiment analysis
def fetch_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Function to implement stop loss
def implement_stop_loss(ticker, quantity, current_price, stop_price):
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=quantity,
            side='sell',
            type='stop',
            stop_price=stop_price,
            time_in_force='gtc'
        )
        logging.info(f'Stop loss order placed: {order}')
    except Exception as e:
        logging.error(f'Error placing stop loss order: {str(e)}')

# Function for portfolio management (example function)
def rebalance_portfolio():
    # Example function to rebalance portfolio based on current holdings and desired allocations
    pass

# Main function
def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers
    start_date = '2023-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    while True:
        try:
            for ticker in tickers:
                # Fetch data
                data = fetch_stock_data(ticker, start_date, end_date)

                # Preprocess data
                scaled_data, scaler = preprocess_data(data)

                # Split data into training and testing sets
                X, y = scaled_data[:, :-1], scaled_data[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train LSTM model
                model = train_lstm_model(X_train, y_train)

                # Generate buy signal based on LSTM prediction
                predicted_price = generate_buy_signal(model, data, scaler)
                current_price = data['Close'].iloc[-1]

                # Calculate RSI
                rsi = calculate_rsi(data)

                # Check breakout resistance
                breakout_resistance = check_breakout_resistance(data)

                # Fetch sentiment analysis (example)
                sentiment_score = fetch_sentiment_analysis("Example news headline or article text")

                # Generate buy signal conditions
                if predicted_price > current_price and rsi < 30 and breakout_resistance:
                    signal_message = f"BUY Signal for {ticker}:\nPredicted Price: {predicted_price}\nCurrent Price: {current_price}\nRSI: {rsi}\nBreakout Resistance: {breakout_resistance}\nSentiment Score: {sentiment_score}"
                    send_sms_notification(signal_message)

                    # Example implementation of stop loss
                    implement_stop_loss(ticker, 10, current_price, current_price * 0.95)  # Implement stop loss at 5% below current price

                # Example portfolio rebalancing function call
                rebalance_portfolio()

        except Exception as e:
            logging.error(f'Error occurred: {str(e)}')

        # Wait for next iteration (e.g., check every hour)
        time.sleep(3600)  # 3600 seconds = 1 hour

if __name__ == "__main__":
    main()
