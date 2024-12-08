import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.pyplot as plt
import torch.nn
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

df = yf.download('NVDQ', start='2024-01-01', end='2024-12-01')

df = df['Close']


df['Lag1'] = df['NVDQ'].shift(1)
df['Lag2'] = df['NVDQ'].shift(2)
df.dropna(inplace=True) 

from sklearn.model_selection import train_test_split

X = df[['Lag1', 'Lag2']]  # Features
y = df['NVDQ']           # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")

last_row = df.iloc[-1]  # Get the most recent row
future_data = {
    'Lag1': last_row['NVDQ'], 
    'Lag2': last_row['Lag1']
}

future_prices = []
for _ in range(5):
    # Convert to DataFrame for prediction
    input_data = pd.DataFrame([future_data])

    # Predict next price
    next_price = model.predict(input_data)[0]
    future_prices.append(next_price)

    # Update lagged features for next iteration
    future_data['Lag2'] = future_data['Lag1']
    future_data['Lag1'] = next_price

print("Predicted Prices for Next 5 Days:", future_prices)

