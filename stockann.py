import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
import os

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%b %d, %Y'))
    data.set_index('Date', inplace=True)
    return data

# Function to create datasets with a longer lookback window
def create_dataset(data, lookback=180):  # Longer lookback of 180 days
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

# Feature Engineering: Add moving averages and other indicators
data['10_day_MA'] = data['Close'].rolling(window=10).mean()
data['30_day_MA'] = data['Close'].rolling(window=30).mean()
data['RSI'] = (data['Close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).sum() /
               data['Close'].diff().abs().rolling(14).sum()) * 100
data.fillna(method='bfill', inplace=True)  # Backfill missing values

# Extract features and target
features = data[['Open', 'High', 'Low', 'Volume', 'Close', '10_day_MA', '30_day_MA', 'RSI']].values

# Separate price and volume scaling
scaler_prices = MinMaxScaler(feature_range=(0, 1))
scaler_volume = MinMaxScaler(feature_range=(0, 1))

scaled_features = features.copy()
scaled_features[:, :5] = scaler_prices.fit_transform(features[:, :5])  # Scale price features
scaled_features[:, 5:] = scaler_volume.fit_transform(features[:, 5:])  # Scale additional features

# Train-test split (85%-15%)
train_size = int(len(scaled_features) * 0.85)
train_features, test_features = scaled_features[:train_size], scaled_features[train_size:]

X_train, y_train = create_dataset(train_features, lookback=180)
X_test, y_test = create_dataset(test_features, lookback=180)

# Model Initialization with regularization and recurrent dropout
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_dropout=0.2),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False, recurrent_dropout=0.2),
    Dense(units=2, kernel_regularizer=tf.keras.regularizers.L2(0.01))  # Two outputs: Open and Close
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mean_squared_error')

# Callbacks
checkpoint = ModelCheckpoint('stock_predictor.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr])

# Predict and evaluate
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler_prices.inverse_transform(
    np.hstack([predicted_stock_price, np.zeros((len(predicted_stock_price), features.shape[1] - 2))])
)[:, [0, -1]]  # Scale back only 'Open' and 'Close'
y_test_original = scaler_prices.inverse_transform(
    np.hstack([y_test, np.zeros((len(y_test), features.shape[1] - 2))])
)[:, [0, -1]]

# Graphs: Same as before
# 1. Actual Stock Prices for the last 5 years
plt.figure(figsize=(12, 6))
plt.plot(data.index[-5 * 252:], data['Close'][-5 * 252:], color='blue', label='Actual Stock Prices (Last 5 Years)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices (Last 5 Years)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Training and Validation Loss
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

# MAE, MSE, and R² Score Calculation (Open only)
mae_open = mean_absolute_error(y_test_original[:, 0], predicted_stock_price[:, 0])
mse_open = mean_squared_error(y_test_original[:, 0], predicted_stock_price[:, 0])
r2_open = r2_score(y_test_original[:, 0], predicted_stock_price[:, 0])

print(f"Mean Absolute Error (MAE) - Open: {mae_open}")
print(f"Mean Squared Error (MSE) - Open: {mse_open}")
print(f"R² Score (Open): {r2_open}")

# Predict the next working day's prices
last_180_days = scaled_features[-180:]
last_180_days = last_180_days.reshape(1, last_180_days.shape[0], last_180_days.shape[1])
predicted_next_day = model.predict(last_180_days)
predicted_next_day_prices = scaler_prices.inverse_transform(
    np.hstack([predicted_next_day, np.zeros((1, features.shape[1] - 2))])
)[:, [0, -1]]

predicted_open_price = predicted_next_day_prices[0, 0]
predicted_close_price = predicted_next_day_prices[0, 1]

# Determine the predicted date
most_recent_date = data.index[-1]
predicted_date = most_recent_date + timedelta(days=1)
if most_recent_date.weekday() == 4:  # If Friday, skip to Monday
    predicted_date += timedelta(days=2)

print(f"Predicted prices for {predicted_date.strftime('%b %d, %Y')}:")
print(f"Opening Price: {predicted_open_price}")
print(f"Closing Price: {predicted_close_price}")
