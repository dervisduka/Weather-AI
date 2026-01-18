import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Ngarkimi i modelit
model = load_model('models/tirana_ensemble_v3.h5', compile=False)
scaler = joblib.load('models/scaler_v3.pkl')

# 2. Marrja e të dhënave të fundit reale
df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)

features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month', 'delta_pres', 'temp_change']
current_batch = df[features].tail(168).values # Merr 7 ditët e fundit

weekly_predictions = []

print("⏳ AI po llogarit parashikimin për 7 ditët e ardhshme (168 orë)...")

# 3. Loop-i Rekursiv
for dita in range(7):
    # Përgatit hyrjen (Scaling & Reshaping)
    scaled_input = scaler.transform(current_batch[-168:])
    X_input = scaled_input.reshape(1, 168, len(features))
    
    # Parashiko 24 orët e ardhshme
    pred_24h_scaled = model.predict(X_input, verbose=0)
    
    # Inverse scaling për të marrë vlerat në Celsius
    dummy = np.zeros((24, len(features)))
    dummy[:, 0] = pred_24h_scaled[0]
    pred_24h_celsius = scaler.inverse_transform(dummy)[:, 0]
    
    # Ruaj parashikimet
    weekly_predictions.extend(pred_24h_celsius)
    
    # PËRDITËSIMI I DRITARES: 
    # Krijojmë rreshta të rinj fiktivë për t'ia dhënë modelit për raundin tjetër
    new_rows = np.zeros((24, len(features)))
    new_rows[:, 0] = pred_24h_celsius # Vendosim temperaturat e parashikuara
    # (Opsionale: mund të simulosh orët e muajin, por për temp mjafton kjo)
    
    current_batch = np.vstack([current_batch, new_rows])

# 4. Vizualizimi i Javës
plt.figure(figsize=(15, 7))
plt.plot(weekly_predictions, color='blue', linewidth=2, label='Parashikimi 7 Ditor')

# Shto vija ndarëse për çdo ditë
for i in range(1, 7):
    plt.axvline(x=i*24, color='gray', linestyle='--', alpha=0.5)
    plt.text(i*24 - 12, max(weekly_predictions), f"Dita {i}", horizontalalignment='center')

plt.title('Parashikimi i Temperaturës në Tiranë - 1 Javë (Ensemble AI)')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Orët (nga tani deri në +168h)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig('plots/weekly_prediction.png')
plt.show()

print(f"✅ Parashikimi u përfundua! Grafiku u ruajt te 'plots/weekly_prediction.png'")