# Data Ingestion Pipeline

## src/get_data.py

**Mission:** Fetches historical hourly weather data for Tirana from the Meteostat API. It locates the nearest weather station, downloads comprehensive data from January 2020 to the present, and saves it as raw CSV data.

**Output:** data/tirana_weather_raw.csv.

## src/analysis.py

**Mission:** Performs initial exploratory data analysis on the raw weather dataset. It checks for missing values across all columns, computes basic statistics for key meteorological variables (temperature, humidity, pressure), and generates a temperature trend visualization for the most recent year.

**Insight:** Identified over 6,000 hours of data gaps in the raw dataset.

## src/preprocessing.py

**Mission:** Conducts comprehensive data cleaning and feature engineering. Removes columns with excessive missing data, fills temporal gaps through hourly reindexing, applies linear interpolation for continuous variables, and creates new time-based features (hour, month, day of week).

**Output:** data/tirana_weather_clean.csv.

## src/verify_data.py

**Mission:** Performs technical quality assurance on the cleaned dataset. Ensures temporal continuity (no missing hours), confirms absence of null values, and validates physical correlations between meteorological variables through correlation analysis.

## src/deep_check.py

**Mission:** Executes scientific validation by decomposing the temperature time series into trend, seasonal, and residual components. Analyzes seasonal patterns and monthly distributions to determine dataset suitability for machine learning applications.

## src/data_ingestion.py

**Mission:** Collects current weather data for multiple Albanian cities using the OpenWeatherMap API. Retrieves real-time meteorological information and appends it to a CSV file for supplementary data collection.

**Note:** This script serves as an alternative data source complementing the historical data from Meteostat.