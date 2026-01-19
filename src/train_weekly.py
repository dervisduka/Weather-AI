import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ensemble_model import build_ensemble_model

# 1. Ngarkimi i të dhënave
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)

features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month', 'delta_pres', 'temp_change']
data = df[features].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 2. Krijimi i dritareve për 1 JAVË (168 orë)
X, y = [], []
lookback = 168  # Shikon 1 javë pas
prediction_window = 168  # Parashikon 1 javë para

for i in range(lookback, len(scaled_data) - prediction_window):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i:i+prediction_window, 0]) # Target: 168 orë temperaturë

X, y = np.array(X), np.array(y)

# Split 80/20
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# 3. Ndërtimi i Modelit (output_steps=168)
model = build_ensemble_model((X.shape[1], X.shape[2]), output_steps=168)

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("🚀 Duke nisur trajnimin për parashikimin 1-javor...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          epochs=40, batch_size=32, callbacks=[lr_reduction, early_stop])

# 4. Ruajtja e modelit të ri
model.save('models/tirana_weekly_v4.h5')
joblib.dump(scaler, 'models/scaler_weekly_v4.pkl')
print("✅ Modeli 1-javor u ruajt si 'tirana_weekly_v4.h5'")