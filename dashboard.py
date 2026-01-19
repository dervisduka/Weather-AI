import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os

# Konfigurimi i faqes
st.set_page_config(page_title="Tirana Weather AI", layout="wide", page_icon="🌦️")

st.title("🌦️ Tirana Weather AI - Dashboard Parashikimi")
st.markdown("Ky dashboard përdor modelin **Ensemble (LSTM + GRU) v4** për të parashikuar temperaturën për 7 ditët e ardhshme.")

# Funksioni për ngarkimin e modelit dhe scaler-it
@st.cache_resource
def load_assets():
    model_path = os.path.join('models', 'tirana_weekly_v4.h5')
    scaler_path = os.path.join('models', 'scaler_weekly_v4.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ Nuk u gjetën skedarët e modelit në folderin 'models/'.")
        return None, None
        
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Funksioni për përpunimin e parashikimit
def get_forecast():
    model, scaler = load_assets()
    if model is None: return None
    
    data_path = os.path.join('data', 'tirana_weather_clean.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Karakteristikat që përdor modeli
    features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month']
    df['delta_pres'] = df['pres'].diff().fillna(0)
    df['temp_change'] = df['temp'].diff().fillna(0)
    full_features = features + ['delta_pres', 'temp_change']
    
    # Marrja e dritares së fundit (168 orë)
    last_window = df[full_features].tail(168).values
    scaled_input = scaler.transform(last_window).reshape(1, 168, len(full_features))
    
    # Parashikimi
    prediction_scaled = model.predict(scaled_input, verbose=0)
    
    # Kthimi në vlerat origjinale (Inverse Transform)
    dummy = np.zeros((168, len(full_features)))
    dummy[:, 0] = prediction_scaled[0]
    prediction_celsius = scaler.inverse_transform(dummy)[:, 0]
    
    # Krijimi i DataFrame për parashikimin
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(hours=i+1) for i in range(168)]
    forecast_df = pd.DataFrame({'Data': forecast_dates, 'Temp': prediction_celsius})
    return forecast_df

# Ekzekutimi kryesor i Dashboard-it
try:
    forecast_df = get_forecast()
    
    if forecast_df is not None:
        # Llogaritja e Min/Max për çdo ditë
        forecast_df['Dita_Data'] = forecast_df['Data'].dt.date
        daily_summary = forecast_df.groupby('Dita_Data')['Temp'].agg(['min', 'max']).reset_index()
        daily_summary.columns = ['Data', 'Temp Min (°C)', 'Temp Max (°C)']

        # 1. Shfaqja e Kartave (Metrics) për 4 ditët e para
        st.subheader("📌 Përmbledhja e Ditëve të Ardhshme")
        cols = st.columns(4)
        for i, row in daily_summary.head(4).iterrows():
            with cols[i]:
                st.metric(
                    label=f"📅 {row['Data'].strftime('%d %b')}", 
                    value=f"{row['Temp Max (°C)']:.1f}°C", 
                    delta=f"Min: {row['Temp Min (°C)']:.1f}°C", 
                    delta_color="normal"
                )

        st.divider()

        # 2. Grafiku Interaktiv me Plotly
        st.subheader("📈 Grafiku i Temperaturës për 1 Javë (168 orë)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['Data'], 
            y=forecast_df['Temp'], 
            mode='lines', 
            name='Temperatura',
            line=dict(color='#007bff', width=3),
            hovertemplate='%{x}<br>Temperatura: %{y:.1f}°C<extra></extra>'
        ))
        
        fig.update_layout(
            hovermode="x unified", 
            template="plotly_white",
            xaxis_title="Data dhe Ora",
            yaxis_title="Gradë Celsius (°C)",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Tabela e Detajuar
        st.subheader("📋 Tabela e Ekstremeve Termike")
        # Formatimi i tabelës për dukje më të mirë
        st.dataframe(
            daily_summary.style.format({'Temp Min (°C)': "{:.1f}", 'Temp Max (°C)': "{:.1f}"}),
            use_container_width=True
        )

except Exception as e:
    st.error(f"⚠️ Ndodhi një gabim gjatë llogaritjes: {e}")
    st.info("Këshillë: Kontrollo nëse të dhënat në 'data/tirana_weather_clean.csv' janë të plota.")