import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# 1. Ngarkimi i modelit të ri V5 dhe Scaler-it përkatës
print("⏳ Duke ngarkuar Modelin V5 dhe të dhënat...")
model = load_model('models/tirana_weekly_v5.h5', compile=False)
scaler = joblib.load('models/scaler_weekly_v5.pkl')
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Përgatitja e kolonave (Duhet të jetë identike me trajnimin)
features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month']
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)
full_features = features + ['delta_pres', 'temp_change']

# 2. Përgatitja e të dhënave për testim
data = df[full_features].values
scaled_data = scaler.transform(data)

X_test, y_test_real = [], []
lookback = 168
prediction_window = 72 # Testojmë 3 ditët e para (72 orë) sipas detyrës

# Marrim kampionet e fundit për testim
for i in range(len(scaled_data) - prediction_window - 100, len(scaled_data) - prediction_window):
    X_test.append(scaled_data[i-lookback:i])
    y_test_real.append(scaled_data[i:i+prediction_window, 0])

X_test = np.array(X_test)
y_test_real = np.array(y_test_real)

# 3. Parashikimi
print("🔮 Duke llogaritur parashikimet...")
y_pred_scaled = model.predict(X_test, verbose=0)

# Kthimi në Celsius
y_pred_rescaled = []
y_real_rescaled = []

for i in range(len(y_pred_scaled)):
    # Inverse për parashikimin (vetëm 72 orët e para)
    dummy_pred = np.zeros((prediction_window, len(full_features)))
    dummy_pred[:, 0] = y_pred_scaled[i][:72]
    y_pred_rescaled.append(scaler.inverse_transform(dummy_pred)[:, 0])
    
    # Inverse për vlerat reale
    dummy_real = np.zeros((prediction_window, len(full_features)))
    dummy_real[:, 0] = y_test_real[i]
    y_real_rescaled.append(scaler.inverse_transform(dummy_real)[:, 0])

# 4. Llogaritja e MAE finale
mae = mean_absolute_error(np.array(y_real_rescaled).flatten(), np.array(y_pred_rescaled).flatten())

print("\n" + "="*40)
print(f"🏆 REZULTATI FINAL I MODELIT V5")
print(f"MAE (Mean Absolute Error): {mae:.2f}°C")
print("="*40)

if mae < 2.0:
    print(f"✅ SUKSES! Modeli është {2.0 - mae:.2f}°C më i saktë se limiti i kërkuar.")
else:
    print("⚠️ Modeli ende nuk e ka kapur limitin, por shifra duhet të jetë më e ulët se V4.")