import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIGURIMI I FAQES ---
st.set_page_config(page_title="Tirana Weather AI v2.1", layout="wide", page_icon="🌦️")

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    # Sigurohu që emrat e skedarëve përputhen me ata që ruajte gjatë trajnimit
    model = load_model('models/tirana_weekly_v5.h5', compile=False)
    scaler = joblib.load('models/scaler_weekly_v5.pkl')
    return model, scaler

# Ngarkojmë asetet
try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Gabim në ngarkimin e modelit: {e}")
    st.stop()

# --- FORECAST LOGIC ---
def get_advanced_forecast():
    # Lexojmë të dhënat e fundit
    df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)
    
    features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month']
    df['delta_pres'] = df['pres'].diff().fillna(0)
    df['temp_change'] = df['temp'].diff().fillna(0)
    full_features = features + ['delta_pres', 'temp_change']
    
    # Përgatitja e hyrjes për modelin (168 orët e fundit)
    last_window = df[full_features].tail(168).values
    scaled_input = scaler.transform(last_window).reshape(1, 168, len(full_features))
    
    # Parashikimi
    prediction_scaled = model.predict(scaled_input, verbose=0)[0]
    
    # Kthimi i vlerave në origjinal
    dummy = np.zeros((168, len(full_features)))
    dummy[:, 0] = prediction_scaled
    # Simulojmë trendet për kolonat e tjera që të llogarisim shiun
    dummy[:, 1] = np.linspace(df['rhum'].iloc[-1], 75, 168) # Hum trend
    dummy[:, 2] = np.linspace(df['pres'].iloc[-1], 1010, 168) # Pres trend
    
    rescaled = scaler.inverse_transform(dummy)
    
    last_date = df.index[-1]
    forecast_df = pd.DataFrame({
        'Data': [last_date + timedelta(hours=i+1) for i in range(168)],
        'Temp': rescaled[:, 0],
        'Hum': rescaled[:, 1],
        'Pres': rescaled[:, 2]
    })
    
    # Logjika e Reshjeve: Përzierje e Lagështisë dhe Presionit
    forecast_df['Rain_Prob'] = ((forecast_df['Hum'] / 100) * (1020 / forecast_df['Pres']) * 60).clip(5, 95)
    
    return forecast_df

# Ekzekutojmë parashikimin
forecast_df = get_advanced_forecast()

# --- DASHBOARD UI ---
st.title("🌦️ Parashikimi i Detajuar - Tirana AI")

# 1. MIN / MAX PER CDO DITE
st.subheader("🗓️ Ekstremet Termike Ditore")

forecast_df['Dita_Emri'] = forecast_df['Data'].dt.strftime('%A')
forecast_df['Dita_Data'] = forecast_df['Data'].dt.strftime('%d %b')

# Grupimi për të gjetur min/max për çdo ditë kalendarike
daily_summary = forecast_df.groupby(['Dita_Emri', 'Dita_Data'], sort=False).agg({
    'Temp': ['min', 'max'],
    'Rain_Prob': 'mean'
}).reset_index()
daily_summary.columns = ['Dita', 'Data_Shkurter', 'Min Temp', 'Max Temp', 'Shi_Mesatar']

# Shfaqja me kolona dinamike
num_days = len(daily_summary)
cols = st.columns(num_days)

for i, row in daily_summary.iterrows():
    with cols[i]:
        st.markdown(f"**{row['Dita']}**")
        st.caption(row['Data_Shkurter'])
        st.write(f"📈 {row['Max Temp']:.1f}°C")
        st.write(f"📉 {row['Min Temp']:.1f}°C")
        st.write(f"☔ {row['Shi_Mesatar']:.0f}%")

st.divider()

# 2. GRAFIKU I PËRMIRËSUAR
st.subheader("📈 Trendi Orar (7 Ditët e Ardhshme)")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_df['Data'], y=forecast_df['Temp'], 
    name="Temp (°C)", line=dict(color='#FF4B4B', width=3)
))

fig.add_trace(go.Bar(
    x=forecast_df['Data'], y=forecast_df['Rain_Prob'], 
    name="Mundësia për Shi (%)", marker_color='#1F77B4', opacity=0.4, yaxis='y2'
))

fig.update_layout(
    yaxis=dict(title="Temperatura (°C)", side="left"),
    yaxis2=dict(title="Mundësia për Shi (%)", side="right", overlaying="y", range=[0, 100]),
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# 3. ALERT SYSTEM (Pika e fundit e detyrës)
st.subheader("⚠️ Alerte të Inteligjencës Artificiale")
tomorrow_min = daily_summary.iloc[0]['Min Temp']
if tomorrow_min < 5:
    st.error(f"Kujdes: Temperatura pritet të bjerë në {tomorrow_min:.1f}°C. Rrezik për ngrirje!")
elif daily_summary.iloc[0]['Shi_Mesatar'] > 70:
    st.warning("Priten reshje të dëndura shiu për ditën e nesërme. Merrni ombrellën!")
else:
    st.success("Nuk parashikohen fenomene ekstreme për 24 orët e ardhshme.")