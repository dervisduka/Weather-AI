import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_data():
    # Ngarkojmë të dhënat
    file_path = 'data/tirana_weather_raw.csv'
    if not os.path.exists(file_path):
        print("Skedari nuk u gjet!")
        return

    df = pd.read_csv(file_path, index_col='time', parse_dates=True)

    print("--- Analiza e të dhënave të Tiranës ---")
    print(f"Numri total i rreshtave: {len(df)}")
    
    # 1. Kontrollojmë sa vlera mungojnë për çdo kolonë
    print("\nVlerat që mungojnë (Missing Values):")
    print(df.isnull().sum())

    # 2. Statistikat bazë (Temperatura, Lagështia, Presioni)
    print("\nStatistikat kryesore:")
    print(df[['temp', 'rhum', 'pres']].describe())

    # 3. Krijojmë një grafik të thjeshtë të temperaturës për vitin e fundit
    plt.figure(figsize=(12, 6))
    df['temp']['2025-01-01':].plot()
    plt.title('Temperatura në Tiranë (Janar 2025 - Sot)')
    plt.xlabel('Data')
    plt.ylabel('Gradë Celsius')
    plt.grid(True)
    
    # Ruajmë grafikun
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/tirana_temp_trend.png')
    print("\nGrafiku u ruajt në: plots/tirana_temp_trend.png")
    plt.show()

if __name__ == "__main__":
    analyze_data()