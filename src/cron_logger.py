import os
import sqlite3
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np

TR_TZ = timezone(timedelta(hours=3))

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/metacast_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            model_gust REAL,
            obs_gust REAL,
            delta_wind REAL,
            model_temp REAL,
            obs_temp REAL,
            delta_temp REAL,
            reliability_score REAL,
            active_hazards TEXT
        )
    """)
    conn.commit()
    conn.close()

def run_hourly_logging():
    init_db()

    # 1. Canlı METAR Verisi (Erhaç LTAT)
    obs_gust = None
    obs_temp = None
    hazards = []

    try:
        url_metar = "https://aviationweather.gov/api/data/metar?ids=LTAT&format=json"
        resp = requests.get(url_metar, timeout=8)
        if resp.status_code == 200 and resp.json():
            data = resp.json()[0]
            wspd_kt = data.get("wspd", 0) or 0
            wgst_kt = data.get("wgst", 0) or wspd_kt
            obs_gust = round(wgst_kt * 1.852, 1)
            obs_temp = data.get("temp", None)
            raw_upper = data.get("rawOb", "").upper()

            if any(x in raw_upper for x in [" GR ", " GS ", "+GR", "+GS"]):
                hazards.append("Dolu")
            if " TS" in raw_upper or "TS " in raw_upper or "VCTS" in raw_upper:
                hazards.append("Oraj")
            if " SN " in raw_upper or "+SN" in raw_upper:
                hazards.append("Kar")
    except Exception as e:
        print(f"METAR Hatası: {e}")

    # 2. Sayısal Model Tahmini (Open-Meteo)
    model_gust = 0.0
    model_temp = 0.0
    try:
        url_model = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 38.4312,
            "longitude": 38.0898,
            "hourly": ["wind_gusts_10m", "temperature_2m"],
            "models": ["ecmwf_ifs", "gfs_seamless", "icon_seamless"],
            "forecast_days": 1,
            "timezone": "Europe/Istanbul",
        }
        resp = requests.get(url_model, params=params, timeout=10)
        if resp.status_code == 200:
            h_data = resp.json()["hourly"]
            df_m = pd.DataFrame(h_data)
            df_m["time"] = pd.to_datetime(df_m["time"])
            cur_h = datetime.now(TR_TZ).hour
            now_idx = cur_h if cur_h < len(df_m) else 0

            ecmwf = df_m["wind_gusts_10m_ecmwf_ifs"].iloc[now_idx]
            gfs = df_m["wind_gusts_10m_gfs_seamless"].iloc[now_idx]
            icon = df_m["wind_gusts_10m_icon_seamless"].iloc[now_idx]
            model_gust = round(float((ecmwf + gfs + icon) / 3.0), 1)
            model_temp = round(float(df_m["temperature_2m_ecmwf_ifs"].iloc[now_idx]), 1)
    except Exception as e:
        print(f"Model Tahmin Hatası: {e}")

    # 3. Sapma ve Güvenilirlik Hesabı
    delta_wind = round(model_gust - obs_gust, 1) if obs_gust is not None else None
    delta_temp = round(model_temp - obs_temp, 1) if obs_temp is not None else None

    if obs_gust is not None and obs_temp is not None:
        penalty = min(50.0, abs(delta_wind) * 2.5) + min(30.0, abs(delta_temp) * 6.0)
        rel_score = max(15.0, round(100.0 - penalty, 1))
    else:
        rel_score = 50.0

    hazards_str = ", ".join(hazards) if hazards else "Sakin"

    # 4. Veritabanına Kayıt
    conn = sqlite3.connect("data/metacast_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO validation_logs 
        (timestamp, mode, model_gust, obs_gust, delta_wind, model_temp, obs_temp, delta_temp, reliability_score, active_hazards)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "Cron / GitHub Action",
        model_gust,
        obs_gust,
        delta_wind,
        model_temp,
        obs_temp,
        delta_temp,
        rel_score,
        hazards_str
    ))
    conn.commit()

    # Kayıt sayısını kontrol et ve gerekirse otomatik eğit
    cursor.execute("SELECT COUNT(*) FROM validation_logs WHERE obs_gust IS NOT NULL")
    total_count = cursor.fetchone()[0]
    conn.close()

    print(f"[{datetime.now(TR_TZ)}] Log kaydedildi. Toplam kayıt: {total_count}")

    if total_count > 0 and total_count % 50 == 0:
        try:
            from train import retrain_from_logs
            success, msg = retrain_from_logs(min_new_records=50)
            print(f"Otomatik Eğitim: {msg}")
        except Exception as e:
            print(f"Yeniden eğitim hatası: {e}")

if __name__ == "__main__":
    run_hourly_logging()
