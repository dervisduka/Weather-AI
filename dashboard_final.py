import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.graph_objects as go

# Fix për importin (sigurohu që ensemble_model.py është në src/)
sys.path.append(os.path.join(os.getcwd(), 'src'))

st.set_page_config(page_title="Tirana Weather AI - Ensemble V6", layout="wide")

@st.cache_resource
def load_models():
    # Ngarkojmë modelin e ri V6 (Ensemble)
    # compile=False është e rëndësishme sepse modelet Ensemble shpesh kërkojnë Custom Objects
    m_t = load_model('models/tirana_weekly_v6.h5', compile=False)
    s_t = joblib.load('models/scaler_weekly_v6.pkl')
    
    # Modelet e reshjeve (po i mbajmë v4 nëse nuk ke trajnuar v6 për to)
    m_p = load_model('models/tirana_precip_v4.h5', compile=False)
    s_p = joblib.load('models/scaler_precip_v4.pkl')
    return m_t, s_t, m_p, s_p

st.title("🏙️ Tirana Weather AI - Parashikimi Ensemble (LSTM + GRU)")

try:
    m_temp, s_temp, m_prcp, s_prcp = load_models()
    df = pd.read_csv('data/tirana_weather_clean.csv', index_col=0, parse_dates=True)
    
    # 1. Përgatitja e Tipareve (Saktësisht si në trajnimin V6)
    df['delta_pres'] = df['pres'].diff().fillna(0)
    df['temp_change'] = df['temp'].diff().fillna(0)
    
    # Lista e plotë e tipareve për modelin e temperaturës
    temp_features = ['temp', 'rhum', 'pres', 'wspd', 'prcp', 'hour', 'month', 'delta_pres', 'temp_change']
    
    last_168 = df.tail(168)
    
    # 2. Skalimi dhe Transformimi i Inputit
    # Kujdes: Scaler duhet të përdorë vetëm kolonat e trajnimit
    in_t_scaled = s_temp.transform(last_168[temp_features].values)
    in_t = in_t_scaled.reshape(1, 168, len(temp_features))
    
    # Inputi për reshjet (v4)
    prcp_features = ['prcp', 'rhum', 'pres', 'temp', 'wspd', 'hour', 'month']
    in_p_scaled = s_prcp.transform(last_168[prcp_features].values)
    in_p = in_p_scaled.reshape(1, 168, len(prcp_features))
    
    # 3. Gjenerimi i parashikimit
    p_t = m_temp.predict(in_t, verbose=0)[0]
    p_p = m_prcp.predict(in_p, verbose=0)[0]
    
    # 4. Invertimi i Skalimit (Inverse Transform manual)
    # Duke qenë se kemi 9 tipare, na duhet vetëm i pari (Temperatura)
    res_t = p_t * (s_temp.data_max_[0] - s_temp.data_min_[0]) + s_temp.data_min_[0]
    res_p = np.clip(p_p * (s_prcp.data_max_[0] - s_prcp.data_min_[0]) + s_prcp.data_min_[0], 0, None)
    
    # 5. Vizualizimi
    dates = [df.index[-1] + timedelta(hours=i+1) for i in range(168)]
    forecast_df = pd.DataFrame({'Koha': dates, 'Temperatura': res_t, 'Reshje': res_p})
    forecast_df['Data'] = forecast_df['Koha'].dt.date

# --- GRAFIKU I TEMPERATURËS (VETËM LINJA E TRENDIT) ---
    st.subheader("🌡️ Trendi i Temperaturës (Ensemble V6)")
    fig_t = go.Figure()
    
    fig_t.add_trace(go.Scatter(
        x=forecast_df['Koha'], 
        y=forecast_df['Temperatura'], 
        line=dict(color='#00FFCC', width=3),
        line_shape='spline', # Kjo e bën linjën më të lakuar dhe organike
        mode='lines',        # Siguron që të jetë vetëm linjë, pa pika (markers)
        name="Temp °C"
    ))
    
    fig_t.update_layout(
        template="plotly_dark", 
        height=450, 
        yaxis_title="Gradë Celsius (°C)",
        xaxis_title="Data dhe Ora",
        hovermode="x unified" # Shfaq vlerat më mirë kur kalon mausin mbi grafik
    )
    
    st.plotly_chart(fig_t, use_container_width=True)

    # --- GRAFIKU I RESHJEVE ---
    st.subheader("🌧️ Parashikimi i Reshjeve")
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=forecast_df['Koha'], y=forecast_df['Reshje'], 
                           marker_color='#33C1FF', name="Reshje mm"))
    fig_p.update_layout(template="plotly_dark", height=400, yaxis_title="Milimetra (mm)")
    st.plotly_chart(fig_p, use_container_width=True)

    st.divider()

    # --- TABELA PËRMBLEDHËSE ---
    st.subheader("📅 Përmbledhja Ditore")
    summary_table = forecast_df.groupby('Data').agg({
        'Temperatura': ['min', 'max'],
        'Reshje': 'sum'
    }).reset_index()
    summary_table.columns = ['Data', 'Min (°C)', 'Max (°C)', 'Reshje (mm)']
    
    st.dataframe(summary_table.style.format({
        'Min (°C)': '{:.1f}',
        'Max (°C)': '{:.1f}',
        'Reshje (mm)': '{:.2f}'
    }), use_container_width=True)

except Exception as e:
    st.error(f"Gabim në ngarkimin e modelit Ensemble V6. Sigurohu që modeli dhe scaleri ekzistojnë. Detajet: {e}")