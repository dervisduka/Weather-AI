import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Ngarkimi i modelit dhe scaler-it me compile=False
print("🔄 Duke ngarkuar modelin Ensemble v3...")
model = load_model('models/tirana_ensemble_v3.h5', compile=False)
scaler = joblib.load('models/scaler_v3.pkl')

# 2. Marrja e të dhënave të fundit
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)

# Llogarisim tiparet ekstra
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)

features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month', 'delta_pres', 'temp_change']
last_168_hours = df[features].tail(168).values

# 3. Preprocessing
scaled_input = scaler.transform(last_168_hours)
X_input = np.reshape(scaled_input, (1, 168, len(features)))

# 4. Predict
print("🧠 AI po gjeneron parashikimin...")
prediction_scaled = model.predict(X_input)

# Inverse scaling (rikthimi në Celsius)
dummy = np.zeros((24, len(features)))
dummy[:, 0] = prediction_scaled[0]
prediction_celsius = scaler.inverse_transform(dummy)[:, 0]

# 5. Plot
plt.figure(figsize=(12, 6))
plt.plot(prediction_celsius, marker='o', color='red', label='Parashikimi AI')
plt.title('Parashikimi i Temperaturës në Tiranë (Ensemble v3)')
plt.ylabel('Gradë Celsius (°C)')
plt.grid(True)
plt.legend()
plt.show()

print("\nParashikimi për 24 orët e ardhshme:")
print(prediction_celsius)