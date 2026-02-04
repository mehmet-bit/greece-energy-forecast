import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import concurrent.futures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
# dotenv'i sadece lokal yedek olarak tutuyoruz
try:
    from dotenv import load_dotenv
    load_dotenv("Back.env")
except:
    pass

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Greece Load Forecast AI", layout="wide")

# ==========================================
# 0. AYARLAR VE GÜVENLİK (SECRETS YÖNETİMİ 🔐)
# ==========================================
# Öncelik Sırası:
# 1. Streamlit Secrets (Cloud veya local .streamlit/secrets.toml)
# 2. Sistem Ortam Değişkenleri (Environment Variables)
# 3. Manuel Giriş (Sidebar)

SYSTEM_KEY = None

try:
    # Önce Streamlit Secrets'a bak (Hata verirse 'except'e atlar)
    if "ENTSOE_API_KEY" in st.secrets:
        SYSTEM_KEY = st.secrets["ENTSOE_API_KEY"]
except FileNotFoundError:
    pass # Secrets dosyası yoksa panik yapma
except Exception:
    pass # Başka bir hata olursa da geç

# Secrets'ta bulamadıysa Environment'a bak
if not SYSTEM_KEY and os.getenv("ENTSOE_API_KEY"):
    SYSTEM_KEY = os.getenv("ENTSOE_API_KEY")

ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
GREECE_DOMAIN = "10YGR-HTSO-----Y"

ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
GREECE_DOMAIN = "10YGR-HTSO-----Y"

# --- Model Yükleme ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("lightgbm_model_Temp.pkl")
        return model
    except Exception as e:
        return None

model = load_model()

# ==========================================
# 1. ENTSO-E Veri Çekme (SAATLİK ZORLAMA)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_entsoe_load(api_key, start_date, end_date):
    if not api_key:
        return pd.DataFrame()
        
    start_str = start_date.strftime("%Y%m%d") + "0000"
    end_str = (end_date + timedelta(days=1)).strftime("%Y%m%d") + "0000"

    params = {
        "securityToken": api_key,
        "documentType": "A65",
        "processType": "A16",
        "outBiddingZone_Domain": GREECE_DOMAIN,
        "periodStart": start_str,
        "periodEnd": end_str
    }

    try:
        response = requests.get(ENTSOE_ENDPOINT, params=params, timeout=30)
        
        if response.status_code != 200:
            if "Missing or invalid security token" in response.text:
                st.error("🚨 ERROR: API Key invalid! Please check secrets.toml or input.")
            else:
                st.error(f"API ERROR: {response.text}")
            return pd.DataFrame()

        root = ET.fromstring(response.content)
        ns = {'ns': root.tag.split('}')[0].strip('{')}
        data = []

        for ts in root.findall(".//ns:TimeSeries", ns):
            for period in ts.findall(".//ns:Period", ns):
                start_time = period.find("ns:timeInterval/ns:start", ns).text
                resolution = period.find("ns:resolution", ns).text
                base_time = pd.to_datetime(start_time, utc=True)
                interval = 60 if resolution == "PT60M" else 15

                for point in period.findall("ns:Point", ns):
                    pos = int(point.find("ns:position", ns).text)
                    qty = float(point.find("ns:quantity", ns).text)
                    dt = base_time + timedelta(minutes=(pos - 1) * interval)
                    data.append({"Datetime": dt, "Load_MW": qty})
        
        df = pd.DataFrame(data)
        
        # --- RESAMPLING (HOURLY) ---
        if not df.empty:
            df = df.set_index("Datetime")
            df = df.resample('1h').mean()
            df = df.reset_index()

        return df

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. Hava Durumu (Standart Saatlik)
# ==========================================
def fetch_single_city(name, lat, lon, start_date, end_date):
    today = datetime.utcnow().replace(tzinfo=None)
    df_list = []
    
    if start_date < today:
        try:
            archive_end = min(end_date, today)
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date.date()}&end_date={archive_end.date()}&hourly=temperature_2m&timezone=Europe%2FAthens"
            df = pd.read_json(url)
            if "hourly" in df:
                df_list.append(pd.DataFrame({"Datetime": pd.to_datetime(df["hourly"]["time"]), name: df["hourly"]["temperature_2m"]}))
        except: pass

    if end_date > today:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&forecast_days=16&timezone=Europe%2FAthens"
            df = pd.read_json(url)
            if "hourly" in df:
                df_f = pd.DataFrame({"Datetime": pd.to_datetime(df["hourly"]["time"]), name: df["hourly"]["temperature_2m"]})
                mask = (df_f["Datetime"] >= pd.to_datetime(start_date)) & (df_f["Datetime"] <= pd.to_datetime(end_date))
                df_list.append(df_f.loc[mask])
        except: pass
        
    if df_list:
        return pd.concat(df_list).drop_duplicates("Datetime")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_greece_temp_parallel(start_date, end_date):
    cities = {"Athens": (37.98, 23.72), "Thessaloniki": (40.64, 22.94), "Patras": (38.25, 21.73), "Heraklion": (35.34, 25.13), "Larisa": (39.64, 22.41)}
    dfs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_single_city, name, lat, lon, start_date, end_date): name for name, (lat, lon) in cities.items()}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if not res.empty: dfs.append(res)
    
    if not dfs: return pd.DataFrame()
    df_merged = dfs[0]
    for df_city in dfs[1:]: df_merged = pd.merge(df_merged, df_city, on="Datetime", how="outer")
    
    df_merged = df_merged.sort_values("Datetime").reset_index(drop=True)
    
    temp_cols = [c for c in df_merged.columns if c != "Datetime"]
    df_merged["Temp_GR_Avg"] = df_merged[temp_cols].mean(axis=1)
    df_merged["Datetime"] = pd.to_datetime(df_merged["Datetime"]).dt.tz_localize(None).dt.tz_localize("UTC")
    return df_merged[["Datetime", "Temp_GR_Avg"]]

# ==========================================
# 3. INTERFACE
# ==========================================
st.markdown("<h1 style='text-align:center; color:#0072CE;'>🇬🇷 Greece Load Forecast AI</h1>", unsafe_allow_html=True)

# --- API KEY MANAGEMENT (Akıllı Seçim) ---
if SYSTEM_KEY:
    # Eğer secrets.toml içinde key varsa, kullanıcıya sormuyoruz.
    API_KEY = SYSTEM_KEY
    st.sidebar.success("✅ API Key Loaded from System (Secure)")
else:
    # Eğer yoksa, manuel giriş kutusunu gösteriyoruz.
    st.sidebar.warning("⚠️ System Key not found.")
    API_KEY = st.sidebar.text_input("🔑 Enter ENTSO-E API Key", type="password")

col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=7))
end_date = col2.date_input("End Date", datetime.now() + timedelta(days=2))

if st.button("🚀 Run Model and Analyze", type="primary"):
    
    if not API_KEY:
        st.error("🚨 Please provide an API Key (via Sidebar or secrets.toml)")
        st.stop()
        
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    with st.spinner("Processing Data..."):
        # 1. Weather
        df_weather = fetch_greece_temp_parallel(start_dt, end_dt)
        
        # 2. Forecast
        if not df_weather.empty and model:
            df_pred = df_weather.copy()
            df_pred["Month"] = df_pred["Datetime"].dt.month
            df_pred["Hour"] = df_pred["Datetime"].dt.hour
            df_pred["Weekday"] = df_pred["Datetime"].dt.weekday
            df_pred["DayofYear"] = df_pred["Datetime"].dt.dayofyear
            df_pred["Season"] = df_pred["Month"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
            
            features = ["Season", "Month", "Hour", "Weekday", "DayofYear", "Temp_GR_Avg"]
            df_pred = df_pred.fillna(0)
            df_pred["Predicted_Load_MW"] = model.predict(df_pred[features])
        else:
            df_pred = pd.DataFrame()

        # 3. Actual Data
        fetch_end = min(end_dt, datetime.utcnow())
        df_actual = fetch_entsoe_load(API_KEY, start_dt, fetch_end)
        
        # 4. Visualization
        if not df_pred.empty:
            if not df_actual.empty:
                df_actual["Datetime"] = pd.to_datetime(df_actual["Datetime"])
                merged = pd.merge(df_actual, df_pred, on="Datetime")
                merged = merged.dropna(subset=["Load_MW", "Predicted_Load_MW"])
                
                if not merged.empty:
                    y_true = merged["Load_MW"]
                    y_pred = merged["Predicted_Load_MW"]
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                    st.markdown("### 🏆 Model Performance")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("R² Score", f"{r2:.2%}")
                    m2.metric("MAE", f"{mae:.0f} MW")
                    m3.metric("RMSE", f"{rmse:.0f} MW")
                    st.divider()

                    # VOLUME ANALYSIS
                    total_actual_mwh = merged["Load_MW"].sum()
                    total_pred_mwh = merged["Predicted_Load_MW"].sum()
                    diff_mwh = total_pred_mwh - total_actual_mwh
                    diff_percent = (diff_mwh / total_actual_mwh) * 100 if total_actual_mwh != 0 else 0

                    st.markdown("### ⚡ Total Energy (Volume - MWh)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Actual Total", f"{total_actual_mwh:,.0f} MWh")
                    c2.metric("Predicted Total", f"{total_pred_mwh:,.0f} MWh", f"{diff_mwh:,.0f} MWh ({diff_percent:.1f}%)", delta_color="inverse")
                    
                    status = "✅ BALANCED"
                    if diff_percent > 2: status = "⚠️ Overestimation"
                    elif diff_percent < -2: status = "⚠️ Underestimation"
                    c3.metric("Status", status)
                    st.divider()

            last_pred = df_pred.iloc[-1]["Predicted_Load_MW"]
            st.metric("Latest Prediction", f"{last_pred:,.0f} MW")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_pred["Datetime"], y=df_pred["Predicted_Load_MW"], mode='lines', name='AI Prediction', line=dict(color='#FF5733', width=2, dash='dash')))
            if not df_actual.empty:
                fig.add_trace(go.Scatter(x=df_actual["Datetime"], y=df_actual["Load_MW"], mode='lines', name='Actual', line=dict(color='#1f77b4', width=2)))

            fig.update_layout(title="Hourly Load Prediction", height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 Data Table"):
                st.dataframe(df_pred.head(24))
        else:
            st.warning("No data available.")

st.sidebar.markdown("---")
st.sidebar.caption("Data Sources: ENTSO-E & Open-Meteo.")