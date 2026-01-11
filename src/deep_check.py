import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def deep_verification():
    # 1. Ngarkimi i të dhënave të pastruara
    if not pd.io.common.file_exists('data/tirana_weather_clean.csv'):
        print("Gabim: Nuk u gjet skedari 'tirana_weather_clean.csv'. Ekzekuto preprocessing.py më parë.")
        return

    df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)
    
    print("--- Duke gjeneruar grafikët e verifikimit të thellë ---")

    # 2. Dekompozimi i Serisë Kohore (Trendi dhe Sezonaliteti)
    # Marrim një kampion prej 240 orësh (10 ditë) për të parë qartë ciklin ditë-natë
    sample_data = df['temp'].head(240) 
    result = seasonal_decompose(sample_data, model='additive', period=24)
    
    plt.figure(figsize=(15, 12))
    
    # Grafiku i Sezonalitetit (Cikli 24-orësh)
    plt.subplot(4, 1, 1)
    plt.plot(result.observed)
    plt.title('1. Të dhënat Origjinale (10 ditët e para)')
    
    plt.subplot(4, 1, 2)
    plt.plot(result.trend)
    plt.title('2. Trendi (Drejtimi i lëvizjes së temperaturës)')
    
    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal)
    plt.title('3. Sezonaliteti (Pulsi i rregullt 24-orësh)')
    
    plt.subplot(4, 1, 4)
    plt.plot(result.resid)
    plt.title('4. Residualet (Zhurma/Anomalitë e papritura)')
    
    plt.tight_layout()
    plt.show()

    # 3. Boxplot-i i Muajve (Verifikimi i Stinëve)
    plt.figure(figsize=(12, 6))
    df['month'] = df.index.month
    sns.boxplot(x='month', y='temp', data=df, palette='coolwarm')
    plt.title('Shpërndarja e Temperaturës sipas Muajve (2020-2026)')
    plt.xlabel('Muaji (1=Janar, 12=Dhjetor)')
    plt.ylabel('Temperatura (°C)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    deep_verification()