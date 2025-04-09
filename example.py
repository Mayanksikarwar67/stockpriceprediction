import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import os

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%b %d, %Y'))
    data.set_index('Date', inplace=True)
    return data

# Function to create datasets
def create_dataset(data, lookback=90):  # Lookback of 90 days
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, [0, -1]])  # Predict 'Open' (index 0) and 'Close' (last column)
    return np.array(X), np.array(y)

# Load the stock data
file_path = input("Enter the CSV file path containing stock data: ")
if not os.path.exists(file_path):
    print("Invalid file path. Exiting.")
    exit()

data = load_data_from_csv(file_path)

# Ensure dataset is sorted by date
data.sort_index(inplace=True)

# Extract features and target
features = data[['Open', 'High', 'Low', 'Volume', 'Close']].values
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(features)

# Train-test split (85%-15%)
train_size = int(len(scaled_features) * 0.85)
train_features, test_features = scaled_features[:train_size], scaled_features[train_size:]

X_train, y_train = create_dataset(train_features, lookback=90)
X_test, y_test = create_dataset(test_features, lookback=90)

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

# Train the model for 100 epochs
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr])

# Predict and evaluate
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler_features.inverse_transform(
    np.hstack([predicted_stock_price, np.zeros((len(predicted_stock_price), features.shape[1] - 2))])
)[:, [0, -1]]  # Scale back only 'Open' and 'Close'
y_test_original = scaler_features.inverse_transform(
    np.hstack([y_test, np.zeros((len(y_test), features.shape[1] - 2))])
)[:, [0, -1]]

# 1. Graph: Actual Stock Prices for the last 5 years
plt.figure(figsize=(12, 6))
plt.plot(data.index[-5*252:], data['Close'][-5*252:], color='blue', label='Actual Stock Prices (Last 5 Years)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices (Last 5 Years)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Graph: Loss during training
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 3. Graph: Actual vs Predicted Stock Prices for the last 60 days
# Using last 60 days from the dataset
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

# MAE Calculation for both Open and Close
mae_open = mean_absolute_error(y_test_original[:, 0], predicted_stock_price[:, 0])
mae_close = mean_absolute_error(y_test_original[:, 1], predicted_stock_price[:, 1])
print(f"Mean Absolute Error (MAE) - Open: {mae_open}")
print(f"Mean Absolute Error (MAE) - Close: {mae_close}")

# Predict the next working day's prices
last_90_days = scaled_features[-90:]
last_90_days = last_90_days.reshape(1, last_90_days.shape[0], last_90_days.shape[1])
predicted_next_day = model.predict(last_90_days)
predicted_next_day_prices = scaler_features.inverse_transform(
    np.hstack([predicted_next_day, np.zeros((1, features.shape[1] - 2))])
)[:, [0, -1]]

predicted_open_price = predicted_next_day_prices[0, 0]
predicted_close_price = predicted_next_day_prices[0, 1]

# Determine the predicted date, skip weekends
most_recent_date = data.index[-1]

# Check if the most recent date is a Friday, and skip to Monday if so
if most_recent_date.weekday() == 4:  # 4 means Friday
    predicted_date = most_recent_date + timedelta(days=3)  # Skip to Monday
else:
    predicted_date = most_recent_date + timedelta(days=1)

# Print the predicted date and prices
print(f"Predicted prices for {predicted_date.strftime('%b %d, %Y')}:")
print(f"Opening Price: {predicted_open_price}")
print(f"Closing Price: {predicted_close_price}")
