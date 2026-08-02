import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

np.random.seed(42)
tf.random.set_seed(42)

print("--- Loading Excel Dataset ---")
df = pd.read_excel('PGCB_date_power_demand.xlsx')

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').set_index('datetime')

TARGET = 'generation_mw'
df_hourly = df[[TARGET]].resample('1h').mean()
df_hourly[TARGET] = df_hourly[TARGET].ffill().bfill()

print(f"Data range: {df_hourly.index.min()} to {df_hourly.index.max()}")
print(f"Total hourly records: {len(df_hourly)}")

# Feature Engineering
df_hourly['hour'] = df_hourly.index.hour
df_hourly['dayofweek'] = df_hourly.index.dayofweek
df_hourly['month'] = df_hourly.index.month
df_hourly['lag_1'] = df_hourly[TARGET].shift(1)      # 1 hour ago
df_hourly['lag_24'] = df_hourly[TARGET].shift(24)    # Same hour yesterday
df_hourly['lag_168'] = df_hourly[TARGET].shift(168)  # Same hour last week
df_hourly['rolling_mean_24'] = df_hourly[TARGET].shift(1).rolling(24).mean()


df_model = df_hourly.dropna().copy()
features = ['hour', 'dayofweek', 'month', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24']

#processed dataframe for Streamlit
df_model.to_csv('processed_pgcb_data.csv')

# Time-Based Train/Test Split (80/20)
split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train, y_train = train_df[features], train_df[TARGET]
X_test, y_test = test_df[features], test_df[TARGET]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

#Model 1: XGBoost (Classical ML)
print("\nTraining XGBoost Model")
xgb_model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

#Model 2: LSTM (Deep Learning)
print("Training LSTM Model")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# Reshape input to 3D tensor: [samples, time steps, features]
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=10, batch_size=64, verbose=1)

lstm_preds_scaled = lstm_model.predict(X_test_lstm)
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten()

# Save combined predictions for the Streamlit dashboard
results_df = test_df.copy()
results_df['XGBoost_Pred'] = xgb_preds
results_df['LSTM_Pred'] = lstm_preds
results_df.to_csv('model_results.csv')

#Evaluation Metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

xgb_mae, xgb_rmse, xgb_mape = compute_metrics(y_test.values, xgb_preds)
lstm_mae, lstm_rmse, lstm_mape = compute_metrics(y_test.values, lstm_preds)

print("\nEVALUATION RESULTS")
print(f"XGBoost -> MAE: {xgb_mae:.2f} MW | RMSE: {xgb_rmse:.2f} MW | MAPE: {xgb_mape:.2f}%")
print(f"LSTM    -> MAE: {lstm_mae:.2f} MW | RMSE: {lstm_rmse:.2f} MW | MAPE: {lstm_mape:.2f}%")

# Static Evaluation Plot
plt.figure(figsize=(15, 6))
plt.plot(test_df.index[-168:], y_test.values[-168:], label='Actual Generation (MW)', color='black')
plt.plot(test_df.index[-168:], xgb_preds[-168:], label='XGBoost Forecast', color='orange', linestyle='--')
plt.plot(test_df.index[-168:], lstm_preds[-168:], label='LSTM Forecast', color='green', linestyle=':')
plt.title('PGCB Generation Forecast vs Actual (Last 7 Days of Test Set)')
plt.xlabel('Date')
plt.ylabel('Generation (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()