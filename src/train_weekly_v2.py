import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 1. Ngarkimi i të dhënave
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Karakteristikat (Features)
features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month']
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)
full_features = features + ['delta_pres', 'temp_change']

data = df[full_features].values

# 2. Skalimi - RUAJMË NJE SCALER TË RI V5
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, 'models/scaler_weekly_v5.pkl')

# 3. Krijimi i sekuencave (168h -> 168h)
X, y = [], []
lookback = 168
forecast = 168

for i in range(lookback, len(scaled_data) - forecast):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i:i+forecast, 0])

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.9)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 4. Arkitektura BIDIRECTIONAL
model = Sequential([
    Input(shape=(lookback, len(full_features))),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(GRU(64)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(forecast)
])

model.compile(optimizer='adam', loss='mae')

# 5. Callback-et
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("🚀 Duke trajnuar Modelin V5 (VERSIONI I RI)...")
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# RUAJTJA ME EMËR TË RI
model.save('models/tirana_weekly_v5.h5')
print("✅ Modeli i ri V5 u ruajt me sukses pa fshirë të vjetrin!")