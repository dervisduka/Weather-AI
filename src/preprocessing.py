import pandas as pd
import os

def clean_data():
    file_path = 'data/tirana_weather_raw.csv'
    df = pd.read_csv(file_path, index_col='time', parse_dates=True)

    # 1. Fshijmë kolonat me shumë mungesa
    cols_to_drop = ['snwd', 'wpgt', 'tsun', 'cldc', 'coco']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. RREGULLIMI I VRIMAVE KOHORE (Reindexing)
    # Krijojmë një kalendar të plotë orë pas ore
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)

    # 3. MBUSHJA E TË DHËNAVE
    # Për temperaturën përdorim interpolim (lidhje pika-pika)
    df['temp'] = df['temp'].interpolate(method='linear')
    df['rhum'] = df['rhum'].interpolate(method='linear')
    df['pres'] = df['pres'].interpolate(method='linear')
    
    # Për reshjet (prcp), nëse mungon, supozojmë që nuk ka rënë shi (0)
    df['prcp'] = df['prcp'].fillna(0)
    
    # Për erën, përdorim vlerën e fundit të njohur (forward fill)
    df['wdir'] = df['wdir'].ffill()
    df['wspd'] = df['wspd'].ffill()

    # 4. Feature Engineering
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek

    # 5. Ruajtja
    output_path = 'data/tirana_weather_clean.csv'
    df.to_csv(output_path)
    
    print("--- Pastrimi i avancuar përfundoi ---")
    print(f"Rreshta totalë tani (pa asnjë vrimë): {len(df)}")
    print(f"Vlera null të mbetura: {df.isnull().sum().sum()}")

if __name__ == "__main__":
    clean_data()