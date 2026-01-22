import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# 1. Ngarkimi i modelit dhe të dhënave
model = load_model('models/tirana_weekly_v4.h5', compile=False)
scaler = joblib.load('models/scaler_weekly_v4.pkl')
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Përgatitja e kolonave
features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month']
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)
full_features = features + ['delta_pres', 'temp_change']

# 2. Krijimi i dritareve të testimit (nga fundi i të dhënave)
data = df[full_features].values
scaled_data = scaler.transform(data)

X_test, y_test_real = [], []
lookback = 168
prediction_window = 72 # Testojmë saktësinë për 3 ditë (72 orë) sipas detyrës

# Marrim 10 pika të ndryshme në kohë për të bërë një mesatare të saktë
for i in range(len(scaled_data) - prediction_window - 50, len(scaled_data) - prediction_window):
    X_test.append(scaled_data[i-lookback:i])
    y_test_real.append(scaled_data[i:i+prediction_window, 0])

X_test = np.array(X_test)
y_test_real = np.array(y_test_real)

# 3. Parashikimi
y_pred_scaled = model.predict(X_test, verbose=0)

# Kthimi në Celsius (duke përdorur vetëm kolonën e temperaturës)
# Ne testojmë vetëm 72 orët e para të parashikimit
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

print("-" * 30)
print(f"📊 REZULTATI I TESTIMIT (MAE):")
print(f"MAE për 3 ditë: {mae:.2f}°C")
print("-" * 30)

if mae < 2.0:
    print("✅ SUKSES: Modeli plotëson kriterin e detyrës (MAE < 2°C)!")
else:
    print("⚠️ Modeli ka nevojë për pak më shumë trajnim për të arritur < 2°C.")