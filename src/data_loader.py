import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

def get_robust_session():
    """Bağlantı kopmalarına karşı 3 kez tekrar deneyen güvenli request oturumu."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_malatya_historical_data():
    print("LTAT (Erhaç METAR İstasyonu) referanslı veri indiriliyor...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    hourly_vars = [
        "temperature_2m",
        "dew_point_2m",  # Zirai don tahmini için EKLENDİ
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "cloud_cover",
    ]
    
    # Koordinatlar LTAT (Erhaç Havalimanı METAR) konumuna çekildi
    params = {
        "latitude": 38.4312,
        "longitude": 38.0898,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "hourly": hourly_vars,
        "timezone": "UTC", # METAR verileri (Z) UTC olduğu için zaman eşzamanlaması adına UTC kullanıldı.
    }
    
    session = get_robust_session()
    
    try:
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Kritik API Hatası: {e}")
        return

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    
    # Malatya Erhaç rakımı (~850m) baz alınarak fiziksel düzeltme EKLENDİ
    STATION_ELEVATION_M = 850.0 
    df["estimated_freezing_level_m"] = (
        STATION_ELEVATION_M + (df["temperature_2m"] / 0.0065)
    ).clip(lower=STATION_ELEVATION_M).round(1)

    # Konvektif Kararsızlık (CAPE benzeri yaklaşım)
    df["convective_risk_index"] = (
        (df["temperature_2m"] * (df["relative_humidity_2m"] / 100.0))
        * (df["wind_gusts_10m"] / (df["wind_speed_10m"] + 1e-5))
    ).round(2)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "historical_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Genişletilmiş veri başarıyla kaydedildi! Toplam {len(df)} saatlik kayıt.")
    print(f"Dosya: {output_path}")

if __name__ == "__main__":
    download_malatya_historical_data()
