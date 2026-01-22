import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Ngarkimi i modelit dhe të dhënave
model = load_model('models/tirana_hybrid_v6.h5', compile=False)
scaler_f = joblib.load('models/scaler_features_v6.pkl')
scaler_t = joblib.load('models/scaler_targets_v6.pkl')
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Përgatitja e Test Set (përdorim 1000 orët e fundit)
features = ['temp', 'prcp', 'rhum', 'pres', 'wspd', 'hour', 'month']
df['temp_roll'] = df['temp'].rolling(window=24).mean().fillna(df['temp'])
full_features = features + ['temp_roll']

test_data = df[full_features].tail(1168) # 1000 + 168 lookback
scaled_test = scaler_f.transform(test_data)

X_test, y_true_raw = [], []
lookback = 168
for i in range(lookback, len(scaled_test) - 168):
    X_test.append(scaled_test[i-lookback:i])
    y_true_raw.append(df[['temp', 'prcp']].iloc[len(df)-1168+i : len(df)-1168+i+168].values)

X_test = np.array(X_test)
y_true_raw = np.array(y_true_raw)

# 2. Parashikimi
print("🔍 Duke kalkuluar parashikimet për test-set...")
y_pred_scaled = model.predict(X_test, verbose=0)

# Denormalizimi
y_pred_rescaled = []
for i in range(len(y_pred_scaled)):
    y_pred_rescaled.append(scaler_t.inverse_transform(y_pred_scaled[i]))
y_pred_rescaled = np.array(y_pred_rescaled)

# 3. Kalkulimi i Metrikave (për temperaturën)
y_true_temp = y_true_raw[:, :, 0].flatten()
y_pred_temp = y_pred_rescaled[:, :, 0].flatten()

mae = mean_absolute_error(y_true_temp, y_pred_temp)
mse = mean_squared_error(y_true_temp, y_pred_temp)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_temp, y_pred_temp)

print("\n" + "="*30)
print("📊 REZULTATET E EVALUIMIT (TEMPERATURA)")
print(f"MAE (Gabimi mesatar): {mae:.2f}°C")
print(f"RMSE: {rmse:.2f}°C")
print(f"R2 Score (Saktësia): {r2:.4f}")
print("="*30)

# 4. Vizualizimi i krahasimit (24 orët e para të parashikimit të fundit)
plt.figure(figsize=(12,6))
plt.plot(y_true_raw[-1, :48, 0], label='Realiteti (Actual)', color='blue', linewidth=2)
plt.plot(y_pred_rescaled[-1, :48, 0], label='Parashikimi (Predicted)', color='red', linestyle='--')
plt.title('Krahasimi: Realitet vs Parashikim (48 orët e ardhshme)')
plt.ylabel('Temperatura °C')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()