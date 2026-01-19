import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# 1. Ngarkimi i modelit dhe scaler-it të ri
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(os.path.join(base_path, 'models', 'tirana_weekly_v4.h5'), compile=False)
scaler = joblib.load(os.path.join(base_path, 'models', 'scaler_weekly_v4.pkl'))

# 2. Marrja e të dhënave të fundit (168 orë)
df = pd.read_csv(os.path.join(base_path, 'data', 'tirana_weather_clean.csv'), index_col=0, parse_dates=True)
df['delta_pres'] = df['pres'].diff().fillna(0)
df['temp_change'] = df['temp'].diff().fillna(0)

features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month', 'delta_pres', 'temp_change']
last_window = df[features].tail(168).values
scaled_input = scaler.transform(last_window).reshape(1, 168, len(features))

# 3. Parashikimi
print("🔮 AI po gjeneron parashikimin 7-ditor...")
prediction_scaled = model.predict(scaled_input, verbose=0)

# Kthejmë vlerat në Celsius
dummy = np.zeros((168, len(features)))
dummy[:, 0] = prediction_scaled[0]
prediction_celsius = scaler.inverse_transform(dummy)[:, 0]

# 4. Vizualizimi Profesional
plt.figure(figsize=(16, 8))
plt.plot(prediction_celsius, color='#007bff', linewidth=2.5, label='Temperatura e Parashikuar')

# Shtojmë vijat ndarëse për ditët dhe emrat e ditëve
for i in range(1, 8):
    plt.axvline(x=i*24, color='red', linestyle='--', alpha=0.2)
    if i < 8:
        plt.text(i*24 - 12, max(prediction_celsius) + 1, f"Dita {i}", 
                 horizontalalignment='center', fontweight='bold', color='#555')

plt.title('Parashikimi i Motit në Tiranë - Grafiku 7-Ditor (Ensemble v4)', fontsize=15)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.xlabel('Orët në vijim', fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(loc='upper right')

# Ruajtja e rezultatit final
output_plot = os.path.join(base_path, 'plots', 'final_weekly_forecast.png')
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"✅ Grafiku final u ruajt te: {output_plot}")