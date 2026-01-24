import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Importojmë modelin e ri
from ensemble_model import build_ensemble_model

# 1. Ngarkimi i të dhënave (Saktësisht si te versioni yt)
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month']
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)
full_features = features + ['delta_pres', 'temp_change']

data = df[full_features].values

# 2. Skalimi
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(scaler, 'models/scaler_weekly_v6.pkl')

# 3. Krijimi i sekuencave
X, y = [], []
lookback, forecast = 168, 168

for i in range(lookback, len(scaled_data) - forecast):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i:i+forecast, 0])

X, y = np.array(X), np.array(y)
split = int(len(X) * 0.9)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 4. Ndërtimi i Modelit Ensemble
model = build_ensemble_model((lookback, len(full_features)), forecast)

# 5. Callback-et
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("🚀 Duke trajnuar Modelin V6 ENSEMBLE (Parallel LSTM & GRU)...")
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# RUAJTJA
model.save('models/tirana_weekly_v6.h5')
print("✅ Modeli Ensemble V6 u ruajt me sukses!")