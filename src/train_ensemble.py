import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ensemble_model import build_ensemble_model

# 1. Load data - Duke përdorur skedarin tënd të ri
print("📊 Duke ngarkuar të dhënat...")
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Krijojmë tiparet ekstra që i duhen modelit ensemble për saktësi
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)

# Përzgjedhim kolonat që ekzistojnë në formatin tënd të ri
features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month', 'delta_pres', 'temp_change']
data = df[features].values

# 2. Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Krijimi i dritareve (Lookback 168h -> Predict 24h)
X, y = [], []
lookback = 168
prediction_window = 24

for i in range(lookback, len(scaled_data) - prediction_window):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i:i+prediction_window, 0]) # Target: Temperatura

X, y = np.array(X), np.array(y)

# Split 80% Train, 20% Test
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# 4. Build Model
model = build_ensemble_model((X.shape[1], X.shape[2]), output_steps=prediction_window)

# Callbacks për trajnim inteligjent
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

print(f"🧠 Duke nisur trajnimin Ensemble (LSTM + GRU)...")
print(f"Input shape: {X_train.shape}")

# 5. Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[lr_reduction, early_stop]
)

# 6. Ruajtja e rezultateve
model.save('models/tirana_ensemble_v3.h5')
joblib.dump(scaler, 'models/scaler_v3.pkl')
print("✅ Modeli i ri u ruajt te 'models/tirana_ensemble_v3.h5'")