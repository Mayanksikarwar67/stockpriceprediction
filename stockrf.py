import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import os

# Fetch data from CSV
def fetch_and_normalize_data(file_path):
    if os.path.exists(file_path):
        print(f"Loading and normalizing data from {file_path}...")
        data = pd.read_csv(file_path)
        if not {'Date', 'Open', 'High', 'Low', 'Volume', 'Close'}.issubset(data.columns):
            raise ValueError("CSV does not contain all required columns: ['Date', 'Open', 'High', 'Low', 'Volume', 'Close']")
        
        # Normalize data
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        features = data[['Open', 'High', 'Low', 'Volume', 'Close']]
        
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        return normalized_features, features, scaler
    else:
        raise FileNotFoundError(f"CSV file not found: {file_path}")

# Create dataset
def create_dataset(data, lookback=60):
    X, y_open, y_close = [], [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i].flatten())  # Flatten for Random Forest input
        y_open.append(data[i, 0])  # 'Open' price
        y_close.append(data[i, -1])  # 'Close' price
    return np.array(X), np.array(y_open), np.array(y_close)

# Define CSV path
csv_file_path = "/Users/mayanksikarwar/Desktop/stock_price_prediction/TSLA_2019-11-24_to_2024-11-24.csv"

# Fetch and normalize data
try:
    normalized_data, original_data, scaler = fetch_and_normalize_data(csv_file_path)
except FileNotFoundError as e:
    print(e)
    exit()

# Prepare dataset
lookback = 60
X, y_open, y_close = create_dataset(normalized_data, lookback)

# Split data
X_train, X_test, y_open_train, y_open_test, y_close_train, y_close_test = train_test_split(
    X, y_open, y_close, test_size=0.2, shuffle=False
)

# Train Random Forest models
rf_open = RandomForestRegressor(n_estimators=100, random_state=42)
rf_close = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training Random Forest for Open prices...")
rf_open.fit(X_train, y_open_train)

print("Training Random Forest for Close prices...")
rf_close.fit(X_train, y_close_train)

# Predict and evaluate
predicted_open = rf_open.predict(X_test)
predicted_close = rf_close.predict(X_test)

# Convert normalized predictions to original prices
predicted_open_original = scaler.inverse_transform(
    np.column_stack([predicted_open] * len(original_data.columns)))[:, 0]
predicted_close_original = scaler.inverse_transform(
    np.column_stack([predicted_close] * len(original_data.columns)))[:, -1]

y_open_original = scaler.inverse_transform(
    np.column_stack([y_open_test] * len(original_data.columns)))[:, 0]
y_close_original = scaler.inverse_transform(
    np.column_stack([y_close_test] * len(original_data.columns)))[:, -1]

mae_open = mean_absolute_error(y_open_original, predicted_open_original)
mae_close = mean_absolute_error(y_close_original, predicted_close_original)

print(f'Mean Absolute Error (Open): {mae_open:.2f} USD')
print(f'Mean Absolute Error (Close): {mae_close:.2f} USD')

# Predict next day's prices
latest_data = normalized_data[-lookback:].flatten().reshape(1, -1)
next_day_open = rf_open.predict(latest_data)[0]
next_day_close = rf_close.predict(latest_data)[0]

# Convert next day's predictions to original prices
next_day_open_original = scaler.inverse_transform(
    np.column_stack([[next_day_open]] * len(original_data.columns)))[:, 0][0]
next_day_close_original = scaler.inverse_transform(
    np.column_stack([[next_day_close]] * len(original_data.columns)))[:, -1][0]

print(f"Predicted Opening Price for the next working day: {next_day_open_original:.2f} USD")
print(f"Predicted Closing Price for the next working day: {next_day_close_original:.2f} USD")

# Plot predictions vs actual
plt.figure(figsize=(14, 7))

# Plot Open prices
plt.subplot(2, 1, 1)
plt.plot(y_open_original, color='blue', label='Actual Open Price')
plt.plot(predicted_open_original, color='red', label='Predicted Open Price')
plt.title('Open Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()

# Plot Close prices
plt.subplot(2, 1, 2)
plt.plot(y_close_original, color='blue', label='Actual Close Price')
plt.plot(predicted_close_original, color='red', label='Predicted Close Price')
plt.title('Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()

plt.tight_layout()
plt.show()
