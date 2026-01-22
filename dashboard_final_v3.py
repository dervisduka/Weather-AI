import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Tirana Weather AI - Final V6", layout="wide", page_icon="🌤️")

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model('models/tirana_hybrid_v6.h5', compile=False)
    scaler_f = joblib.load('models/scaler_features_v6.pkl')
    scaler_t = joblib.load('models/scaler_targets_v6.pkl')
    return model, scaler_f, scaler_t

model, scaler_f, scaler_t = load_assets()

# --- FORECAST LOGIC ---
def get_final_forecast():
    df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)
    
    # Feature Engineering (e njëjtë si te trajnimi)
    features = ['temp', 'prcp', 'rhum', 'pres', 'wspd', 'hour', 'month']
    df['temp_roll'] = df['temp'].rolling(window=24).mean().fillna(df['temp'])
    full_features = features + ['temp_roll']
    
    last_window = df[full_features].tail(168).values
    scaled_input = scaler_f.transform(last_window).reshape(1, 168, len(full_features))
    
    # Parashikimi (Output: 1, 168, 2)
    prediction_scaled = model.predict(scaled_input, verbose=0)
    
    # Denormalizimi i targeteve (Temp dhe Prcp)
    # Reshape në (168, 2) për inversion
    reshaped_pred = prediction_scaled.reshape(168, 2)
    rescaled_targets = scaler_t.inverse_transform(reshaped_pred)
    
    last_date = df.index[-1]
    forecast_df = pd.DataFrame({
        'Data': [last_date + timedelta(hours=i+1) for i in range(168)],
        'Temp': rescaled_targets[:, 0],
        'Precip': rescaled_targets[:, 1].clip(0, None) # Mos lejo mm negative
    })
    
    return forecast_df

forecast_df = get_final_forecast()

# --- UI DESIGN ---
st.title("🏙️ Parashikimi Hibrid (LSTM+GRU) - Tirana")
st.markdown(f"**Përditësimi i fundit:** {forecast_df['Data'].iloc[0].strftime('%d %B, %H:%M')}")

# 1. KARTAT DITORE (MIN/MAX)
forecast_df['Dita'] = forecast_df['Data'].dt.strftime('%A, %d %b')
daily = forecast_df.groupby('Dita', sort=False).agg({
    'Temp': ['min', 'max'],
    'Precip': 'sum'
}).reset_index()
daily.columns = ['Data', 'Min', 'Max', 'Total_mm']

cols = st.columns(len(daily))
for i, row in daily.iterrows():
    with cols[i]:
        st.metric(row['Data'].split(',')[0], f"{row['Max']:.1f}°C", f"{row['Min']:.1f}°C", delta_color="inverse")
        st.caption(f"🌧️ {row['Total_mm']:.1f} mm")

st.divider()

# 2. GRAFIKU KRYESOR (Dual Axis)
st.subheader("📊 Analiza e Parashikimit 7-Ditor")
fig = go.Figure()

# Linja e Temperaturës
fig.add_trace(go.Scatter(x=forecast_df['Data'], y=forecast_df['Temp'], 
                         name="Temp (°C)", line=dict(color='#FF4B4B', width=3)))

# Bar-at e Reshjeve (mm)
fig.add_trace(go.Bar(x=forecast_df['Data'], y=forecast_df['Precip'], 
                     name="Reshje (mm)", marker_color='#00CCFF', yaxis='y2', opacity=0.7))

fig.update_layout(
    yaxis=dict(title="Temperatura (°C)"),
    yaxis2=dict(title="Reshje (mm)", side="right", overlaying="y", range=[0, max(forecast_df['Precip'])+5]),
    hovermode="x unified",
    template="plotly_dark",
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig, use_container_width=True)

# 3. ALERTE AUTOMATIKE
st.subheader("⚠️ Njoftime nga Sistemi")
tomorrow_precip = daily.iloc[0]['Total_mm']
if tomorrow_precip > 10:
    st.warning(f"Reshje të dendura të parashikuara: {tomorrow_precip:.1f} mm. Kujdes nga përmbytjet lokale!")
elif daily.iloc[0]['Min'] < 2:
    st.error(f"Temperaturat në rënie ({daily.iloc[0]['Min']:.1f}°C). Rrezik ngrirjeje!")
else:
    st.success("Kushte atmosferike të qëndrueshme për 24 orët e ardhshme.")