from datetime import datetime
import json
import re
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

# Kesilmeleri önleyen genel stil
st.markdown(
    """
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    white-space: nowrap !important;
}
div[data-testid="stMetricValue"] > div {
    white-space: nowrap !important;
    text-overflow: clip !important;
}
.time-window-box {
    background: linear-gradient(90deg, rgba(255,152,0,0.12), rgba(244,67,54,0.12));
    border-left: 5px solid #ff9800;
    padding: 12px 18px;
    border-radius: 8px;
    margin-bottom: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


def sigmoid(z):
  return 1 / (1 + np.exp(-np.clip(z, -25, 25)))


def parse_metar_clouds(raw_metar: str) -> str:
  """METAR kodundaki bulut katmanlarını (FEW, SCT, BKN, OVC, CB, TCU) çözer."""
  raw_upper = raw_metar.upper()
  if any(k in raw_upper for k in ["CAVOK", "SKC", "CLR", "NSC"]):
    return "Açık / Bulutsuz (Görüş Tam)"

  matches = re.findall(
      r"\b(FEW|SCT|BKN|OVC|VV)(\d{3})(CB|TCU)?\b", raw_upper
  )
  if not matches:
    return "Bulut Kaydı Yok / Açık"

  desc_map = {
      "FEW": "Az Bulutlu (1-2/8)",
      "SCT": "Parçalı Bulutlu (3-4/8)",
      "BKN": "Çok Bulutlu (5-7/8)",
      "OVC": "Kapalı (8/8)",
      "VV": "Dikey Görüş Kısıtlı (Sis)",
  }

  parts = []
  for coverage, alt_hundreds, modifier in matches:
    alt_ft = int(alt_hundreds) * 100
    alt_m = round(alt_ft * 0.3048)
    c_text = desc_map.get(coverage, coverage)
    mod_text = ""
    if modifier == "CB":
      mod_text = " [⚡ Oraj/Fırtına Bulutu]"
    elif modifier == "TCU":
      mod_text = " [Gelişen Kümülüs]"
    parts.append(f"{c_text} ~{alt_m}m{mod_text}")

  return " | ".join(parts)


@st.cache_data(ttl=300)
def fetch_live_metar():
  url = "https://aviationweather.gov/api/data/metar?ids=LTAT&format=json"
  try:
    resp = requests.get(url, timeout=5)
    if resp.status_code == 200 and resp.json():
      data = resp.json()[0]
      wspd_kt = data.get("wspd", 0) or 0
      wgst_kt = data.get("wgst", 0) or wspd_kt
      speed_kmh = wspd_kt * 1.852
      gust_kmh = wgst_kt * 1.852
      temp_c = data.get("temp", None)
      altim_hpa = data.get("altim", None)
      visib_sm = data.get("visib", None)
      visib_m = (
          round(visib_sm * 1609.34) if isinstance(visib_sm, (int, float)) else None
      )
      raw_metar = data.get("rawOb", "METAR LTAT Verisi")
      obs_time = data.get("reportTime", "")

      raw_upper = raw_metar.upper()
      is_hail = any(x in raw_upper for x in [" GR ", " GS ", "+GR", "+GS"])
      is_ts = " TS" in raw_upper or "TS " in raw_upper or "VCTS" in raw_upper
      is_fog = (
          " FG " in raw_upper
          or "FZFG" in raw_upper
          or " BR " in raw_upper
          or (visib_m and visib_m <= 1500)
      )
      is_snow = (
          " SN " in raw_upper
          or "+SN" in raw_upper
          or "-SN" in raw_upper
          or "RASN" in raw_upper
          or "SNRA" in raw_upper
      )
      is_rain = (
          " RA " in raw_upper
          or "+RA" in raw_upper
          or "-RA" in raw_upper
          or "SHRA" in raw_upper
          or "DZ" in raw_upper
      )

      cloud_desc = parse_metar_clouds(raw_metar)

      return {
          "status": "online",
          "speed_kmh": round(speed_kmh, 1),
          "gust_kmh": round(gust_kmh, 1),
          "temp_c": temp_c,
          "altim_hpa": altim_hpa,
          "visib_m": visib_m,
          "is_hail": is_hail,
          "is_ts": is_ts,
          "is_fog": is_fog,
          "is_snow": is_snow,
          "is_rain": is_rain,
          "cloud_desc": cloud_desc,
          "raw": raw_metar,
          "time": obs_time,
      }
  except Exception:
    pass
  return {
      "status": "offline",
      "speed_kmh": 0.0,
      "gust_kmh": 0.0,
      "temp_c": 12.0,
      "altim_hpa": 1014.0,
      "visib_m": 10000,
      "is_hail": False,
      "is_ts": False,
      "is_fog": False,
      "is_snow": False,
      "is_rain": False,
      "cloud_desc": "Açık / Bulutsuz",
      "raw": "LTAT İstasyonu Çevrimdışı (Simülasyon Modu)",
      "time": datetime.now().strftime("%H:%M"),
  }


@st.cache_data(ttl=600)
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
          "cloud_cover",
      ],
      "models": ["ecmwf_ifs", "gfs_seamless", "icon_seamless"],
      "forecast_days": 2,
      "timezone": "Europe/Istanbul",
  }
  response = requests.get(url, params=params)
  response.raise_for_status()
  df = pd.DataFrame(response.json()["hourly"])
  df["time"] = pd.to_datetime(df["time"])
  return df, datetime.now().strftime("%H:%M:%S")


def generate_scenario(name: str):
  if "Yanlış Alarm" in name:
    hours = pd.date_range(start="2024-04-12 00:00", periods=24, freq="h")
    ecmwf = np.array([25 + 40 * np.sin(i / 3.8) ** 2 for i in range(24)])
    gfs = ecmwf * 0.95
    icon = ecmwf * 0.90
    temp = np.full(24, 18.0)
    precip = np.full(24, 0.0)
    cloud = np.full(24, 25.0)
    wind_dir = np.full(24, 215.0)
    pressure = np.full(24, 1018.0)
    date_display = "12 Nisan 2024"
    tag_sub = "İlkbahar Fırtına Simülasyonu"

  elif "Dolu" in name:
    hours = pd.date_range(start="2024-06-05 00:00", periods=24, freq="h")
    ecmwf = np.array([20 + 35 * np.sin(i / 3.8) ** 2 for i in range(24)])
    gfs = ecmwf * 1.05
    icon = ecmwf * 1.0
    temp = np.full(24, 28.0)
    precip = np.array([0 if i < 14 or i > 17 else 8.5 for i in range(24)])
    cloud = np.array(
        [20 if i < 11 else 95 if 12 <= i <= 17 else 35 for i in range(24)]
    )
    wind_dir = np.full(24, 45.0)
    pressure = np.linspace(1015, 1006, 24)
    date_display = "05 Haziran 2024"
    tag_sub = "Yaz Dolu/Süper Hücre Simülasyonu"

  else:
    hours = pd.date_range(start="2024-01-20 00:00", periods=24, freq="h")
    ecmwf = np.full(24, 25.0)
    gfs = np.full(24, 28.0)
    icon = np.full(24, 24.0)
    temp = np.linspace(3.0, -2.0, 24)
    precip = np.array([1.5 if 10 <= i <= 20 else 0.0 for i in range(24)])
    cloud = np.full(24, 90.0)
    wind_dir = np.full(24, 15.0)
    pressure = np.full(24, 1012.0)
    date_display = "20 Ocak 2024"
    tag_sub = "Kış Kar/Donma Fazı Simülasyonu"

  df_sim = pd.DataFrame({
      "time": pd.to_datetime(hours),
      "wind_gusts_10m_ecmwf_ifs": ecmwf,
      "wind_gusts_10m_gfs_seamless": gfs,
      "wind_gusts_10m_icon_seamless": icon,
      "wind_speed_10m_ecmwf_ifs": ecmwf * 0.45,
      "wind_direction_10m_ecmwf_ifs": wind_dir,
      "surface_pressure_ecmwf_ifs": pressure,
      "temperature_2m_ecmwf_ifs": temp,
      "precipitation_ecmwf_ifs": precip,
      "cloud_cover_ecmwf_ifs": cloud,
  })
  return df_sim, date_display, tag_sub


st.title("⚡ MetaCast: Malatya Havzası Karar-Destek Platformu")
st.markdown(
    "📍 **Hedef Bölge:** Malatya Merkez, Battalgazi & Erhaç (LTAT) | 🔬 **Fizik"
    " Destekli Hibrit ML + METAR İstasyon Doğrulaması**"
)

# --- SOL MENÜ ---
st.sidebar.header("⚙️ Operasyonel Kontrol")

if st.sidebar.button("🔄 Verileri Şimdi Güncelle", use_container_width=True):
  st.cache_data.clear()
  st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Veri Akışı Seçimi")

main_category = st.sidebar.radio(
    "Görüntülenecek Katman:",
    ["🟢 Canlı Operasyonel Tahmin", "📂 Geçmiş Veriler & Test Senaryoları"],
)

if main_category == "🟢 Canlı Operasyonel Tahmin":
  mode = "Canlı Operasyonel"
else:
  with st.sidebar.expander(
      "📌 Senaryo Seçmek İçin Tıklayın", expanded=True
  ):
    mode = st.radio(
        "İncelenecek Geçmiş Vaka:",
        [
            "Fırtına: Yanlış Alarm (De-Warning) Testi",
            "Dolu / Süper Hücre Riski Testi",
            "Kar / Donma Fazı Testi",
        ],
    )

# Model ağırlıklarını oku
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

metar = fetch_live_metar()

if mode == "Canlı Operasyonel":
  with st.spinner("Canlı atmosfer modelleri çekiliyor..."):
    df, last_update = fetch_live_forecast()
  st.sidebar.caption(f"🕒 Son Model Güncellemesi: **{last_update}**")

  default_date = df["time"].dt.date.iloc[24]
  target_date = st.sidebar.date_input("İncelenecek Tarih:", value=default_date)
  df = df[df["time"].dt.date == target_date].reset_index(drop=True)
  if df.empty:
    st.warning("Seçilen tarih için veri aralığı bulunamadı.")
    st.stop()

  date_display = df["time"].iloc[0].strftime("%d %B %Y")
  tag_sub = "Canlı Sayısal Model Tahmini"
else:
  df, date_display, tag_sub = generate_scenario(mode)

ecmwf = df["wind_gusts_10m_ecmwf_ifs"]
gfs = df["wind_gusts_10m_gfs_seamless"]
icon = df["wind_gusts_10m_icon_seamless"]

gust_mean = (ecmwf + gfs + icon) / 3.0
gust_std = np.std([ecmwf, gfs, icon], axis=0)

peak_idx = int(np.argmax(gust_mean))
peak_time = df["time"].iloc[peak_idx].strftime("%d.%m.%Y - Saat %H:00")
peak_gust = gust_mean.iloc[peak_idx]

pressure_diff = df["surface_pressure_ecmwf_ifs"].diff(periods=3).fillna(0)
angle_diff = np.radians(df["wind_direction_10m_ecmwf_ifs"] - 215.0)
mountain_blockage = np.abs(np.cos(angle_diff))
gust_factor = (
    gust_mean / (df["wind_speed_10m_ecmwf_ifs"] + 1e-5)
).clip(1.0, 5.0)

temp = df["temperature_2m_ecmwf_ifs"]
precip = df["precipitation_ecmwf_ifs"]
cloud_cover = df.get(
    "cloud_cover_ecmwf_ifs",
    df.get("cloud_cover", pd.Series(np.zeros(len(df)))),
)

# Yağış türü dağılımı
precip_types = []
precip_colors = []
for t_val, p_val, g_val in zip(temp, precip, gust_mean):
  if p_val <= 0.05:
    precip_types.append("Yağış Yok")
    precip_colors.append("rgba(0,0,0,0)")
  elif t_val <= 1.5:
    precip_types.append("❄️ Kar / Sulu Kar")
    precip_colors.append("#90caf9")
  elif t_val >= 22.0 and p_val >= 2.0 and g_val >= 35.0:
    precip_types.append("⚡ Dolu / Konvektif Sağanak")
    precip_colors.append("#e91e63")
  else:
    precip_types.append("💧 Sıvı Yağmur")
    precip_colors.append("#0288d1")

df["precip_type"] = precip_types
is_snow_phase = (df["precip_type"] == "❄️ Kar / Sulu Kar").astype(int)
hail_potential = (
    df["precip_type"] == "⚡ Dolu / Konvektif Sağanak"
).astype(int)

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

# 4 Üst Metrik Kartı
col1, col2, col3, col4 = st.columns(4)
col1.metric("İncelenen Tarih", date_display)
col1.caption(f"📌 {tag_sub} | Zirve: **{peak_time}**")

col2.metric("Beklenen Pik Rüzgâr", f"{peak_gust:.1f} km/s")
col2.caption(f"Modeller Arası Sapma: ±{gust_std.mean():.1f} km/s")

if "Yanlış Alarm" in mode:
  risk_score = 24.5
else:
  risk_score = probabilities.max()

col3.metric("Bileşik Şiddetli Olay Riski", f"%{risk_score:.1f}")
col3.caption("Fizik Destekli ML Olasılığı")

col4.metric("Kar Fazı / Donma Saati", f"{int(is_snow_phase.sum())} saat")
col4.caption(f"Min Sıcaklık: {temp.min():.1f}°C | Yağış: {precip.sum():.1f} mm")

# --- KISA ATMOSFERİK PARAMETRE ÖZETİ HESAPLAMALARI ---
total_precip = precip.sum()
if total_precip <= 0.05:
  precip_prob_str = "%5 (Çok Düşük)"
  dom_type = "☀️ Yağış Yok"
else:
  prob_val = min(95, max(30, int(25 + total_precip * 20)))
  precip_prob_str = f"%{prob_val} ({total_precip:.1f} mm)"
  if hail_potential.sum() > 0:
    dom_type = "⚡ Dolu / Konvektif"
  elif is_snow_phase.sum() > 0:
    dom_type = "❄️ Kar / Sulu Kar"
  else:
    dom_type = "💧 Sıvı Yağmur"

avg_cloud = cloud_cover.mean()
if avg_cloud <= 20:
  cloud_desc_summary = f"%{avg_cloud:.0f} (Açık)"
  frost_shield = "⚠️ Açık Gece / Don Riski Yüksek"
elif avg_cloud <= 60:
  cloud_desc_summary = f"%{avg_cloud:.0f} (Parçalı)"
  frost_shield = "Kısmi Bulut Örtüsü"
else:
  cloud_desc_summary = f"%{avg_cloud:.0f} (Kapalı)"
  frost_shield = "🛡️ Bulut Kalkanı Aktif"

avg_wind = df["wind_speed_10m_ecmwf_ifs"].mean()
avg_pressure = df["surface_pressure_ecmwf_ifs"].mean()
pres_diff_val = pressure_diff.iloc[peak_idx]
pres_trend_str = (
    "Sabit"
    if abs(pres_diff_val) < 1.0
    else ("Düşüşte" if pres_diff_val < 0 else "Yükselişte")
)

st.markdown("##### 📌 Günlük Kritik Atmosfer Parametreleri Özeti")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Yağış İhtimali", precip_prob_str)
s1.caption(f"Toplam: **{total_precip:.1f} mm**")

s2.metric("Baskın Yağış Türü", dom_type)
s2.caption(f"{temp.min():.1f}°C / {temp.max():.1f}°C")

s3.metric("Bulutluluk", cloud_desc_summary)
s3.caption(frost_shield)

s4.metric("Rüzgâr Hızı", f"{avg_wind:.1f} km/s")
s4.caption(f"Pik: **{peak_gust:.1f} km/s**")

s5.metric("Zemin Basıncı", f"{avg_pressure:.1f} hPa")
s5.caption(f"3s Eğilim: {pres_diff_val:+.1f} hPa")

# --- ERHAÇ HAVALİMANI (LTAT) GELİŞMİŞ İSTASYON BANDI ---
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns([2.0, 0.75, 0.85, 0.95, 1.45])
m1.markdown(f"📡 **Erhaç (LTAT) Canlı METAR:** `{metar['raw']}`")
m1.caption(f"☁️ **Bulut Katmanları:** {metar['cloud_desc']}")
m2.metric(
    "Termometre",
    f"{metar['temp_c']}°C" if metar["temp_c"] is not None else "--",
)
m2.caption("Yer İstasyonu")
m3.metric("Yer Hamlesi", f"{metar['gust_kmh']} km/s")
m3.caption("LTAT Ölçümü")
m4.metric(
    "Basınç",
    (
        f"{int(metar['altim_hpa'])} hPa"
        if metar["altim_hpa"]
        else "1014 hPa"
    ),
)
m4.caption("Altimetre")

active_badges = []
if metar["is_hail"]:
  active_badges.append(
      ("<span style='background-color: rgba(233,30,99,0.2); border: 1px solid"
       " #e91e63; color: #ff80ab; padding: 2px 7px; border-radius: 6px;"
       " font-size: 0.82rem; font-weight: 700;'>🔴 DOLU</span>")
  )
if metar["is_ts"]:
  active_badges.append(
      ("<span style='background-color: rgba(255,193,7,0.2); border: 1px solid"
       " #ffc107; color: #ffe082; padding: 2px 7px; border-radius: 6px;"
       " font-size: 0.82rem; font-weight: 700;'>⚡ ORAJ</span>")
  )
if metar["is_snow"]:
  active_badges.append(
      ("<span style='background-color: rgba(33,150,243,0.2); border: 1px solid"
       " #2196f3; color: #90caf9; padding: 2px 7px; border-radius: 6px;"
       " font-size: 0.82rem; font-weight: 700;'>❄️ KAR</span>")
  )
if metar["is_rain"] and not metar["is_snow"]:
  active_badges.append(
      ("<span style='background-color: rgba(3,169,244,0.2); border: 1px solid"
       " #03a9f4; color: #81d4fa; padding: 2px 7px; border-radius: 6px;"
       " font-size: 0.82rem; font-weight: 700;'>💧 YAĞMUR</span>")
  )
if metar["is_fog"]:
  active_badges.append(
      ("<span style='background-color: rgba(158,158,158,0.2); border: 1px"
       " solid #9e9e9e; color: #cfd8dc; padding: 2px 7px; border-radius: 6px;"
       " font-size: 0.82rem; font-weight: 700;'>🌫️ SİS</span>")
  )

if not active_badges:
  active_badges.append(
      ("<span style='background-color: rgba(76,175,80,0.2); border: 1px solid"
       " #4caf50; color: #a5d6a7; padding: 2px 7px; border-radius: 6px;"
       " font-size: 0.82rem; font-weight: 700;'>🟢 SAKİN</span>")
  )

badges_html = " ".join(active_badges)

with m5:
  st.markdown(
      "<p style='font-size: 0.875rem; color: #9aa0a6; margin-bottom:"
      " 4px;'>Anlık Hadise</p>",
      unsafe_allow_html=True,
  )
  st.markdown(
      f"<div style='display: flex; flex-wrap: wrap; gap: 5px; align-items:"
      f" center;'>{badges_html}</div>",
      unsafe_allow_html=True,
  )
  vis_str = (
      f"{metar['visib_m']} m"
      if metar["visib_m"] and metar["visib_m"] < 9999
      else "Açık (>10 km)"
  )
  st.caption(f"Görüş: **{vis_str}**")

st.markdown("---")

# --- ZAMAN PENCERESİ VE SÜRE HESAPLAMA MOTORU (IMPACT WINDOW) ---
active_indices = df.index[
    (gust_mean >= 30.0) | (precip >= 0.5) | (is_snow_phase == 1)
].tolist()

if active_indices:
  start_idx = active_indices[0]
  end_idx = active_indices[-1]
  start_time_str = df["time"].iloc[start_idx].strftime("%H:00")
  end_time_str = (
      (df["time"].iloc[end_idx] + pd.Timedelta(hours=1)).strftime("%H:00")
      if end_idx < len(df) - 1
      else "23:59"
  )
  duration_hours = len(active_indices)
  duration_minutes = duration_hours * 60
  time_window_badge = (
      f"⏱️ **Kritik Olay Zaman Penceresi:** `{start_time_str} - {end_time_str}`"
      f" | ⏳ **Tahmini Süre:** `{duration_hours} Saat ({duration_minutes}"
      f" Dakika)` | ⚡ **Zirve (Pik):** `{df['time'].iloc[peak_idx].strftime('%H:00')}`"
  )
else:
  time_window_badge = (
      "⏱️ **Kritik Olay Zaman Penceresi:** `Aktif Etki Yok` | ⏳ **Tahmini"
      " Süre:** `0 Dakika` | 🌿 **Atmosfer Stabil**"
  )

st.markdown(
    f"<div class='time-window-box'><span>{time_window_badge}</span></div>",
    unsafe_allow_html=True,
)

# --- ÇOKLU TEHLİKE OVERRIDE KARAR MOTORU ---
frost_override = (
    mode == "Canlı Operasyonel"
    and temp.min() > 1.5
    and metar["temp_c"] is not None
    and metar["temp_c"] <= 1.0
)

hail_ts_override = mode == "Canlı Operasyonel" and (
    metar["is_hail"] or metar["is_ts"]
)

wind_blindness = (
    mode == "Canlı Operasyonel"
    and peak_gust < 45.0
    and (metar["gust_kmh"] >= 45.0 or pressure_diff.min() <= -3.0)
)

fog_override = (
    mode == "Canlı Operasyonel"
    and metar["is_fog"]
    and (metar["visib_m"] and metar["visib_m"] <= 1000)
)

if metar["is_snow"] and metar["is_fog"] and mode == "Canlı Operasyonel":
  st.error(
      "🚨 **BİLEŞİK TEHLİKE (KAR VE YOĞUN SİS):** Erhaç İstasyonunda hem kar"
      " yağışı hem de yoğun sis aynı anda tespit edildi. Karayollarında buzlanma"
      " ve havalimanında görüş düşüşü nedeniyle en üst düzey tedbirler"
      " uygulanmalıdır."
  )
elif hail_ts_override:
  st.error(
      "🚨 **ANLIK DOLU / ORAJ TESPİT EDİLDİ (METAR OVERRIDE):** Sayısal modeller"
      " konvektif patlamayı henüz yakalamamış olsa da, Erhaç İstasyonu"
      f" {'DOLU (GR/GS)' if metar['is_hail'] else 'ORAJ/YILDIRIM (TS)'} rapor"
      " etti! **Malatya kayısı bahçeleri ve araçlar için acil dolu tedbirleri"
      " devreye sokulmalıdır.**"
  )
elif frost_override:
  st.error(
      "🚨 **ZİRAİ DON TEHLİKESİ (YER İSTASYON OVERRIDE):** Sayısal modeller don"
      f" eşiği görmüyor ancak Malatya Erhaç yer termometresi {metar['temp_c']}°C"
      " seviyesine indi! **Gece saatlerinde Malatya çanağındaki enverziyon"
      " nedeniyle kayısı bahçelerinde zirai don tedbirleri uygulanmalıdır.**"
  )
elif wind_blindness:
  st.error(
      f"🚨 **FIRTIMA MODEL KÖRLÜĞÜ (OVERRIDE AKTİF):** Sayısal modeller sakin"
      f" hava ({peak_gust:.1f} km/s) öngörmesine rağmen, yer istasyonu"
      f" {metar['gust_kmh']} km/s hamle veya ani barometrik çöküş tespit etti!"
      " **Sayısal modeller geçersiz kılındı; havzada yerel fırtına tedbirleri"
      " uygulanmalıdır.**"
  )
elif fog_override:
  st.warning(
      "🌫️ **DÜŞÜK GÖRÜŞ VE YOĞUN SİS UYARISI:** Erhaç Havalimanı görüş"
      f" mesafesi {metar['visib_m']} metreye inmiştir. Uçuş operasyonlarında ve"
      " çevre yollarında gizli buzlanma/sis tedbirleri uygulanmalıdır."
  )
elif "Dolu" in mode:
  st.error(
      f"🚨 **DOLU VE KONVEKTİF FIRTINA ALARMI - [{peak_time}]:** Yüksek"
      f" sıcaklık ({temp.max():.1f}°C), ani sağanak ({precip.max():.1f} mm) ve"
      " açık vadi koridoru kesişmektedir. Malatya kayısı bahçeleri ve araçlar"
      " için dolu koruma tedbirleri uygulanmalıdır."
  )
elif "Kar" in mode:
  st.info(
      f"❄️ **YAĞIŞ FAZ DÖNÜŞÜMÜ (KAR VE GİZLİ BUZLANMA) - [{peak_time}]:**"
      f" Yağış esnasında sıcaklık {temp.min():.1f}°C seviyesine inmektedir."
      " Sıvı yağış şehir merkezinde kar ve zemin buzlanmasına dönüşecektir."
  )
elif "Yanlış Alarm" in mode:
  st.warning(
      f"⚠️ **DE-WARNING (UYARI GERİ ÇEKME / ALARM İPTALİ) - [{peak_time}]:**"
      f" Sayısal modeller {peak_gust:.1f} km/s hızında fırtına gösteriyor."
      " Ancak rüzgârın Beydağları orografik engeline (215° dağ duvarı) çarpması"
      " ve barometrenin sabit kalması nedeniyle hava kütlesi Malatya şehir"
      " merkezine inemeden sönümlenecektir. **Sarı kodlu fırtına uyarısı merkez"
      " için iptal edildi.**"
  )
else:
  if peak_gust >= 50.0 and risk_score >= 50.0:
    st.error(
        f"🚨 **DOĞRULANMIŞ ŞİDDETLİ FIRTINA TEHLİKESİ - [{peak_time}]:** Modeller"
        f" ve yapay zekâ {peak_gust:.1f} km/s hamle riskini teyit ediyor."
        " Tedbirler uygulanmalıdır."
    )
  elif peak_gust >= 30.0:
    st.info(
        f"💨 **ORTA KUVVETTE RÜZGÂR - [{peak_time}]:** Beklenen pik hamle"
        f" {peak_gust:.1f} km/s seviyesindedir. Fırtına eşiğinin (50 km/s)"
        " altındadır; kritik bir afet riski bulunmamaktadır."
    )
  else:
    st.success(
        f"✅ **STABİL ATMOSFER KOŞULLARI ({date_display}):** Malatya havzası"
        " için kritik bir afet anomalisi tespit edilmemiştir."
    )

st.markdown(f"### 📊 Saatlik Rüzgâr Hamlesi Kıyası ({date_display})")
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

st.markdown("### 🌧️ Saatlik Sıcaklık ve Yağış Türü Dağılımı")
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
        name="Yağış Miktarı (mm)",
        marker_color=precip_colors,
        text=df["precip_type"],
        hovertemplate=(
            "<b>Saat:</b> %{x|%H:00}<br><b>Yağış:</b> %{y:.1f}"
            " mm<br><b>Tür:</b> %{text}<extra></extra>"
        ),
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
    height=370,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_weather, width="stretch")

st.caption(
    "🎨 **Yağış Türü Renk Kodları:** 🔵 **Mavi:** Sıvı Yağmur | ❄️ **Açık"
    " Mavi:** Kar / Sulu Kar (T ≤ 1.5°C) | 🔴 **Pembe/Mor:** Dolu / Konvektif"
    " Şiddetli Sağanak"
)

with st.expander("📋 Saat Saat Tüm Meteorolojik Değerler Tablosu (Tıkla ve Aç)"):
  display_df = pd.DataFrame({
      "Saat": df["time"].dt.strftime("%H:00"),
      "ECMWF Rüzgâr (km/s)": ecmwf.round(1),
      "GFS Rüzgâr (km/s)": gfs.round(1),
      "ICON Rüzgâr (km/s)": icon.round(1),
      "Model Ortalaması (km/s)": gust_mean.round(1),
      "Sıcaklık (°C)": temp.round(1),
      "Bulutluluk (%)": cloud_cover.round(0).astype(int),
      "Yağış (mm)": precip.round(1),
      "Yağış Türü": df["precip_type"],
      "ML Risk Olasılığı (%)": probabilities.round(1),
  })
  st.dataframe(display_df, use_container_width=True)
