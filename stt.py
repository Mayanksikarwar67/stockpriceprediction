import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import yfinance as yf

# Function to fetch stock data using yfinance
def fetch_data_from_yfinance(ticker, start_date):
    print(f"Fetching data for {ticker} from {start_date} to today...")
    stock_data = yf.download(ticker, start=start_date)
    if stock_data.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range.")
    return stock_data

# Function to create datasets
def create_dataset(data, lookback=90):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, [0, -1]])  # Predict 'Open' (index 0) and 'Close' (last column)
    return np.array(X), np.array(y)

# Get current date and calculate the start date for the last 5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# User inputs
ticker = input("Enter the stock ticker (e.g., TSLA): ")
lookback = 90

# Fetch stock data
data = fetch_data_from_yfinance(ticker, start_date.strftime('%Y-%m-%d'))

# Prepare the dataset
data = data[['Open', 'High', 'Low', 'Volume', 'Close']]
features = data.values
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(features)

# Train-test split (85%-15%)
train_size = int(len(scaled_features) * 0.85)
train_features, test_features = scaled_features[:train_size], scaled_features[train_size:]

X_train, y_train = create_dataset(train_features, lookback)
X_test, y_test = create_dataset(test_features, lookback)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Model Initialization
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(units=100, return_sequences=True),
    Dropout(0.3),
    LSTM(units=50),
    Dropout(0.3),
    Dense(units=2)  # Two outputs: Open and Close
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
checkpoint = ModelCheckpoint('stock_predictor.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr])

# Predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler_features.inverse_transform(
    np.hstack([predicted_stock_price, np.zeros((len(predicted_stock_price), features.shape[1] - 2))])
)[:, [0, -1]]  # Scale back only 'Open' and 'Close'
y_test_original = scaler_features.inverse_transform(
    np.hstack([y_test, np.zeros((len(y_test), features.shape[1] - 2))])
)[:, [0, -1]]

# Plotting
# 1. Actual Stock Prices for the last 5 years
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], color='blue', label='Actual Stock Prices (Last 5 Years)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices (Last 5 Years)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Loss during training
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 3. Actual vs Predicted Stock Prices for the last 60 days
last_60_actual = data['Close'].iloc[-60:].values
last_60_predicted = predicted_stock_price[-60:]
last_60_dates = data.index[-60:]

plt.figure(figsize=(12, 6))
plt.plot(last_60_dates, last_60_actual, label='Actual Closing Price', color='red')
plt.plot(last_60_dates, last_60_predicted[:, 1], label='Predicted Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Closing Prices (Last 60 Days)')
plt.legend()
plt.grid(True)
plt.show()

# Metrics
mae_open = mean_absolute_error(y_test_original[:, 0], predicted_stock_price[:, 0])
mae_close = mean_absolute_error(y_test_original[:, 1], predicted_stock_price[:, 1])
mse_open = mean_squared_error(y_test_original[:, 0], predicted_stock_price[:, 0])
mse_close = mean_squared_error(y_test_original[:, 1], predicted_stock_price[:, 1])
r2_open = r2_score(y_test_original[:, 0], predicted_stock_price[:, 0])
r2_close = r2_score(y_test_original[:, 1], predicted_stock_price[:, 1])

print(f"Mean Absolute Error (MAE) - Open: {mae_open}")
print(f"Mean Absolute Error (MAE) - Close: {mae_close}")
print(f"Mean Squared Error (MSE) - Open: {mse_open}")
print(f"Mean Squared Error (MSE) - Close: {mse_close}")
print(f"R² Score (Open): {r2_open}")
print(f"R² Score (Close): {r2_close}")

# Next day's prediction
last_90_days = scaled_features[-90:]
last_90_days = last_90_days.reshape(1, last_90_days.shape[0], last_90_days.shape[1])
predicted_next_day = model.predict(last_90_days)
predicted_next_day_prices = scaler_features.inverse_transform(
    np.hstack([predicted_next_day, np.zeros((1, features.shape[1] - 2))])
)[:, [0, -1]]

predicted_open_price = predicted_next_day_prices[0, 0]
predicted_close_price = predicted_next_day_prices[0, 1]

# Predicted date
most_recent_date = data.index[-1]
predicted_date = most_recent_date + timedelta(days=1)
if most_recent_date.weekday() == 4:  # Friday
    predicted_date += timedelta(days=2)  # Skip to Monday

print(f"Predicted prices for {predicted_date.strftime('%b %d, %Y')}:")
print(f"Opening Price: {predicted_open_price}")
print(f"Closing Price: {predicted_close_price}")
