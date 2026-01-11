import requests
import pandas as pd
import os
from datetime import datetime

class WeatherDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cities = ["Tirana", "Durres", "Vlore", "Shkoder", "Korce"]

    def get_current_weather(self, city):
        params = {
            'q': f"{city},AL",
            'appid': self.api_key,
            'units': 'metric' 
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': city,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description']
            }
        except Exception as e:
            print(f"Gabim gjatë marrjes së të dhënave për {city}: {e}")
            return None

    def save_to_csv(self, data_list, filename="data/raw_weather.csv"):
        df = pd.DataFrame(data_list)
        # Nëse skedari ekziston, shto të dhënat e reja (append), përndryshe krijoje të ri
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Të dhënat u ruajtën me sukses në {filename}")

# Shembull përdorimi
# API_KEY = "VENDOS_API_KEY_TUAJ_KETU"
# collector = WeatherDataCollector(API_KEY)
# weather_list = [collector.get_current_weather(city) for city in collector.cities]
# collector.save_to_csv([w for w in weather_list if w is not None])