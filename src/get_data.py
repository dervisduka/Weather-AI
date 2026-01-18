import pandas as pd
from datetime import datetime
import os
import meteostat    

def fetch_albania_weather():
    # 1. Konfigurimi
    meteostat.config.block_large_requests = False
# Krijo pikën gjeografike për Tiranën
    tirana_point = meteostat.Point(41.3275, 19.8187, 110)    # 2. Gjejmë stacionin më të afërt (përdorim shkronja të vogla)
    print("--- Duke kërkuar stacionin më të afërt ---")
    

    station = meteostat.stations.nearby(tirana_point, limit=1)

    if station is None or station.empty:
        print("Gabim: Nuk u gjet asnjë stacion meteorologjik.")
        return

    station_id = station.index[0]
    station_name = station['name'].values[0]
    print(f"Stacioni i gjetur: {station_id} - {station_name}")

    start = datetime(2020, 1, 1)
    end = datetime.now()

    try:
        # 3. Shkarkojmë të dhënat (hourly me të vogël)
        print(f"Duke shkarkuar të dhënat për {station_name}...")
        data = meteostat.hourly(station_id, start, end)
        df = data.fetch()

        if df is not None and not df.empty:
            # Rregullimi i folderit data
            # Gjen folderin ku ndodhet skripti (src)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Ngjitet një nivel lart te rrënja e projektit (Weather-AI)
            base_dir = os.path.dirname(current_dir)

            # Bashkon rrugën me folderin 'data'
            output_dir = os.path.join(base_dir, 'data')

            # Sigurohet që folderi ekziston
            os.makedirs(output_dir, exist_ok=True)

            # Rruga finale e skedarit                        
            file_path = os.path.join(output_dir, 'tirana_weather_raw.csv')
            df.to_csv(file_path)
            
            print(f"\nSUKSES! U shkarkuan {len(df)} rreshta.")
            print(f"Skedari u ruajt këtu: {file_path}")
            print("\nPamja e fundit e të dhënave:")
            print(df.tail())
        else:
            print("Gabim: Serveri nuk ktheu të dhëna (None ose Empty).")

    except Exception as e:
        print(f"Ndodhi një gabim gjatë shkarkimit: {e}")

if __name__ == "__main__":
    fetch_albania_weather()