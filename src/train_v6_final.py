import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. NGARKIMI DHE PËRGATITJA E TË DHËNAVE
# ==========================================
print("📂 Duke ngarkuar të dhënat për modelin Hibrid...")
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Karakteristikat (Features)
features = ['temp', 'prcp', 'rhum', 'pres', 'wspd', 'hour', 'month']
# Feature Engineering: Mesatare lëvizëse 24-orëshe për stabilitet
df['temp_roll'] = df['temp'].rolling(window=24).mean().fillna(df['temp'])
full_features = features + ['temp_roll']

# Objektivat (Targets): Temperatura dhe Reshjet në mm
targets = ['temp', 'prcp']

# Skalimi (MinMaxScaler)
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

scaled_features = scaler_features.fit_transform(df[full_features])
scaled_targets = scaler_targets.fit_transform(df[targets])

X, y = [], []
lookback = 168  # 1 javë histori
forecast = 168  # 1 javë parashikim

for i in range(lookback, len(df) - forecast):
    X.append(scaled_features[i-lookback:i])
    y.append(scaled_targets[i:i+forecast])

X, y = np.array(X), np.array(y)

# ==========================================
# 2. NDËRTIMI I MODELIT HIBRID (LSTM + GRU)
# ==========================================
print("🏗️ Duke ndërtuar arkitekturën Hibride: Bidirectional LSTM + GRU...")

inputs = Input(shape=(lookback, X.shape[2]))

# Shtresa 1: Bidirectional LSTM (Kujtesa afatgjatë)
x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = Dropout(0.2)(x)

# Shtresa 2: Bidirectional GRU (Efikasiteti për ndryshime të shpejta)
x = Bidirectional(GRU(64, return_sequences=False))(x)
x = Dropout(0.2)(x)

# Dalja: 168 orë * 2 variabla = 336 neurone
x = Dense(forecast * 2)(x)

# Reshape për të pasur formatin (168, 2) -> [Temp, Prcp]
outputs = Reshape((forecast, 2))(x)

model = Model(inputs=inputs, outputs=outputs)

# Optimizimi për MAE (Mean Absolute Error)
model.compile(optimizer='adam', loss='mae', metrics=['mse'])

model.summary()

# ==========================================
# 3. TRAJNIMI I MODELIT
# ==========================================
print("🚀 Duke nisur trajnimin e modelit V6 Hibrid...")

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X, y, 
    epochs=60, 
    batch_size=32, 
    validation_split=0.15, 
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 4. RUAJTJA E REZULTATEVE
# ==========================================
print("💾 Duke ruajtur modelin dhe skalerët e rinj...")

model.save('models/tirana_hybrid_v6.h5')
joblib.dump(scaler_features, 'models/scaler_features_v6.pkl')
joblib.dump(scaler_targets, 'models/scaler_targets_v6.pkl')

print("✅ Trajnimi përfundoi!")
print(f"📊 Gabimi mesatar (MAE) në trajnim: {history.history['loss'][-1]:.4f}")