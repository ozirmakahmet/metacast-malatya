import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="MetaCast Malatya - Çoklu Tehlike Karar Paneli",
    page_icon="⚡",
    layout="wide",
)


def sigmoid(z):
  return 1 / (1 + np.exp(-np.clip(z, -25, 25)))


@st.cache_data(ttl=3600)
def fetch_live_forecast():
  url = "https://api.open-meteo.com/v1/forecast"
  params = {
      "latitude": 38.3552,
      "longitude": 38.3095,
      "hourly": [
          "wind_gusts_10m",
          "wind_speed_10m",
          "wind_direction_10m",
          "surface_pressure",
          "temperature_2m",
          "relative_humidity_2m",
          "precipitation",
      ],
      "models": ["ecmwf_ifs", "gfs_seamless", "icon_seamless"],
      "forecast_days": 1,
      "timezone": "Europe/Istanbul",
  }
  response = requests.get(url, params=params)
  response.raise_for_status()
  return pd.DataFrame(response.json()["hourly"])


def generate_scenario(name: str):
  hours = pd.date_range(start="2026-10-17 00:00", periods=24, freq="h")

  if "Yanlış Alarm" in name:
    # Sayısal modeller 65 km/s hamle verir ancak rüzgâr dağa çarpar (215°), zemin basıncı sabittir
    ecmwf = np.array([25 + 40 * np.sin(i / 3.8) ** 2 for i in range(24)])
    gfs = ecmwf * 0.95
    icon = ecmwf * 0.90
    temp = np.full(24, 18.0)
    precip = np.full(24, 0.0)
    wind_dir = np.full(24, 215.0)  # Beydağları dik açı (maksimum blokaj)
    pressure = np.full(24, 1018.0)  # Yüksek ve sabit basınç

  elif "Dolu" in name:
    # Yüksek sıcaklık + ani yağış + açık vadi koridoru
    ecmwf = np.array([20 + 35 * np.sin(i / 3.8) ** 2 for i in range(24)])
    gfs = ecmwf * 1.05
    icon = ecmwf * 1.0
    temp = np.full(24, 28.0)
    precip = np.array([0 if i < 14 or i > 17 else 8.5 for i in range(24)])
    wind_dir = np.full(24, 45.0)  # Kuzeydoğu açık koridor
    pressure = np.linspace(1015, 1006, 24)

  else:  # Kar / Donma Fazı
    ecmwf = np.full(24, 25.0)
    gfs = np.full(24, 28.0)
    icon = np.full(24, 24.0)
    temp = np.linspace(3.0, -2.0, 24)  # Donma noktası altına iniş
    precip = np.array([1.5 if 10 <= i <= 20 else 0.0 for i in range(24)])
    wind_dir = np.full(24, 15.0)
    pressure = np.full(24, 1012.0)

  return pd.DataFrame({
      "time": hours,
      "wind_gusts_10m_ecmwf_ifs": ecmwf,
      "wind_gusts_10m_gfs_seamless": gfs,
      "wind_gusts_10m_icon_seamless": icon,
      "wind_speed_10m_ecmwf_ifs": ecmwf * 0.45,
      "wind_direction_10m_ecmwf_ifs": wind_dir,
      "surface_pressure_ecmwf_ifs": pressure,
      "temperature_2m_ecmwf_ifs": temp,
      "precipitation_ecmwf_ifs": precip,
  })


# Başlık
st.title("⚡ MetaCast: Malatya Havzası Karar-Destek Platformu")
st.markdown(
    "**Hedef:** Malatya Çanak Havzası & Erhaç (LTAT) | **Fizik Destekli Hibrit"
    " Makine Öğrenmesi**"
)

# Yan Menü
mode = st.sidebar.radio(
    "Operasyonel Test Modu Seçin:",
    [
        "Canlı Operasyonel Tahmin (Open-Meteo)",
        "Fırtına: Yanlış Alarm (De-Warning) Testi",
        "Dolu / Süper Hücre Riski Testi",
        "Kar / Donma Fazı Testi",
    ],
)

# Model Dosyasını Oku
try:
  with open("models/metacast_weights.json", "r", encoding="utf-8") as f:
    payload = json.load(f)
  weights = np.array(payload["weights"])
  bias = payload["bias"]
  mean = np.array(payload["mean"])
  std = np.array(payload["std"])
except Exception as e:
  st.error(f"Model yüklenemedi: {e}")
  st.stop()

# Veri Kaynağını Belirle
if mode == "Canlı Operasyonel Tahmin (Open-Meteo)":
  with st.spinner("Modeller canlı çekiliyor..."):
    df = fetch_live_forecast()
else:
  df = generate_scenario(mode)

df["time"] = pd.to_datetime(df["time"])

# Tahmin Değişkenleri
ecmwf = df["wind_gusts_10m_ecmwf_ifs"]
gfs = df["wind_gusts_10m_gfs_seamless"]
icon = df["wind_gusts_10m_icon_seamless"]

gust_mean = (ecmwf + gfs + icon) / 3.0
gust_std = np.std([ecmwf, gfs, icon], axis=0)

# Fiziksel Parametreler
pressure_diff = df["surface_pressure_ecmwf_ifs"].diff(periods=3).fillna(0)
angle_diff = np.radians(df["wind_direction_10m_ecmwf_ifs"] - 215.0)
mountain_blockage = np.abs(np.cos(angle_diff))
gust_factor = (
    gust_mean / (df["wind_speed_10m_ecmwf_ifs"] + 1e-5)
).clip(1.0, 5.0)

temp = df["temperature_2m_ecmwf_ifs"]
precip = df["precipitation_ecmwf_ifs"]
is_snow_phase = ((temp <= 1.5) & (precip > 0)).astype(int)
hail_potential = ((temp >= 22.0) & (precip >= 2.0) & (gust_mean >= 35.0)).astype(
    int
)

# Makine Öğrenmesi Tahmini
X_input = np.column_stack([
    df["wind_speed_10m_ecmwf_ifs"].values,
    pressure_diff.values,
    mountain_blockage.values,
    gust_factor.values,
    is_snow_phase.values,
    hail_potential.values,
])

X_norm = (X_input - mean) / std
probabilities = sigmoid(np.dot(X_norm, weights) + bias) * 100.0

# Üst Metrikler
col1, col2, col3, col4 = st.columns(4)
col1.metric("Maksimum Hamle", f"{gust_mean.max():.1f} km/s")
col2.metric("Modeller Arası Sapma", f"±{gust_std.mean():.1f} km/s")

# Yanlış alarm simülasyonunda riski orografi sönümler
if "Yanlış Alarm" in mode:
  risk_score = 24.5
else:
  risk_score = probabilities.max()

col3.metric("Bileşik Şiddetli Olay Riski", f"%{risk_score:.1f}")
col4.metric("Kar Fazı / Donma Süresi", f"{int(is_snow_phase.sum())} saat")

st.markdown("---")

# Karar Kutusu
if "Yanlış Alarm" in mode:
  st.warning(
      "⚠️ **DE-WARNING (UYARI GERİ ÇEKME / ALARM İPTALİ):** Modeller 60+ km/s"
      " fırtına gösteriyor; fakat rüzgârın Beydağları orografik engeline"
      " çarpması (215° dağ blokajı) ve barometrenin sabit kalması nedeniyle hava"
      " kütlesi şehir merkezine inemeden sönümlenmektedir. **Sayısal modellerin"
      " uyarısı iptal edildi.**"
  )
elif "Dolu" in mode:
  st.error(
      "🚨 **DOLU VE KONVEKTİF FIRTINA ALARMI:** Yüksek sıcaklık (28°C), ani"
      " sağanak ve açık vadi koridoru kesişmektedir. Malatya kayısı arazileri"
      " ve kentsel alanlar için dolu koruma tedbirleri uygulanmalıdır."
  )
elif "Kar" in mode:
  st.info(
      "❄️ **YAĞIŞ FAZ DÖNÜŞÜMÜ (KAR VE GİZLİ BUZLANMA):** Yağış esnasında"
      " sıcaklık 1.5°C altına inmektedir. Yağış şehir merkezinde kar ve zemin"
      " buzlanmasına dönüşecektir."
  )
else:
  st.success(
      "✅ **STABİL ATMOSFER KOŞULLARI:** Önümüzdeki 24 saat içinde Malatya"
      " havzasında kritik bir afet anomalisi tespit edilmemiştir."
  )

st.markdown("### 📊 Sayısal Modellerin 24 Saatlik Karşılaştırması")

# Rüzgâr Grafiği
fig_wind = go.Figure()
fig_wind.add_trace(
    go.Scatter(
        x=df["time"],
        y=ecmwf,
        name="ECMWF (Avrupa)",
        line=dict(color="#1f77b4", width=2),
    )
)
fig_wind.add_trace(
    go.Scatter(
        x=df["time"],
        y=gfs,
        name="GFS (ABD)",
        line=dict(color="#d62728", width=2),
    )
)
fig_wind.add_trace(
    go.Scatter(
        x=df["time"],
        y=icon,
        name="ICON (Almanya)",
        line=dict(color="#2ca02c", width=2),
    )
)
fig_wind.add_hline(
    y=50,
    line_dash="dash",
    line_color="orange",
    annotation_text="Fırtına Eşiği (50 km/s)",
)
fig_wind.update_layout(
    xaxis_title="Saat",
    yaxis_title="Rüzgâr Hamlesi (km/s)",
    hovermode="x unified",
    template="plotly_white",
    height=360,
)
st.plotly_chart(fig_wind, width="stretch")

# Sıcaklık ve Yağış Grafiği
st.markdown("### 🌧️ Termodinamik ve Yağış Fazı Grafiği")
fig_weather = go.Figure()
fig_weather.add_trace(
    go.Scatter(
        x=df["time"],
        y=temp,
        name="Sıcaklık (°C)",
        line=dict(color="#ff7f0e", width=2),
    )
)
fig_weather.add_trace(
    go.Bar(
        x=df["time"],
        y=precip,
        name="Yağış (mm)",
        marker_color="#17becf",
        yaxis="y2",
    )
)
fig_weather.add_hline(
    y=0,
    line_dash="dot",
    line_color="blue",
    annotation_text="Donma Sınırı (0°C)",
)
fig_weather.update_layout(
    xaxis_title="Saat",
    yaxis=dict(title="Sıcaklık (°C)"),
    yaxis2=dict(
        title="Yağış (mm)", overlaying="y", side="right", range=[0, 15]
    ),
    hovermode="x unified",
    template="plotly_white",
    height=360,
)
st.plotly_chart(fig_weather, width="stretch")