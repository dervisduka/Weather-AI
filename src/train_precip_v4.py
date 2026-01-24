import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ensemble_model import build_ensemble_model

# 1. Ngarkimi i të dhënave
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Tiparet që ndikojnë më shumë te shiu: Lagështia, Presioni dhe Era
features = ['prcp', 'rhum', 'pres', 'temp', 'wspd', 'hour', 'month']
data = df[features].values

# Përdorim skaler të veçantë për këtë model
scaler_precip = MinMaxScaler()
scaled_data = scaler_precip.fit_transform(data)

X, y = [], []
lookback = 168       # 1 javë histori
prediction_window = 168  # 1 javë parashikim reshjesh

for i in range(lookback, len(scaled_data) - prediction_window):
    X.append(scaled_data[i-lookback:i])
    # Target: kolona 0 (prcp) për 168 orët e ardhshme
    y.append(scaled_data[i:i+prediction_window, 0])

X, y = np.array(X), np.array(y)

# Split 80/20
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# 2. Ndërtimi i Modelit Ensemble (Specializuar për Reshje)
# Përdorim arkitekturën tënde me output_steps=168
model_precip = build_ensemble_model((X.shape[1], X.shape[2]), output_steps=168)

# Për reshjet, përdorim një patience më të lartë sepse janë më të vështira për t'u mësuar
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🌧️ Duke nisur trajnimin e ekspertit për Reshjet (1-javor)...")
model_precip.fit(X_train, y_train, validation_data=(X_test, y_test), 
                 epochs=10, batch_size=32, callbacks=[lr_reduction, early_stop])

# 3. Ruajtja e modelit
model_precip.save('models/tirana_precip_v4.h5')
joblib.dump(scaler_precip, 'models/scaler_precip_v4.pkl')
print("✅ Modeli i reshjeve u ruajt si 'tirana_precip_v4.h5'")