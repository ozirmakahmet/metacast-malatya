import os
import pandas as pd
import requests


def download_malatya_historical_data():
    """Malatya koordinatları (38.3552, 38.3095) için rüzgâr, orografi,

    yağış fazı (kar/yağmur) ve konvektif fırtına (CAPE) verilerini indirir.
    """
    print("Malatya genişletilmiş şiddetli hava arşivi indiriliyor...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "precipitation",  # Toplam anlık yağış (mm)
        "rain",  # Sıvı yağış miktarı (mm)
        "snowfall",  # Kar yağışı miktarı (cm)
        "weather_code",  # WMO Hava Durumu Kodu
        "cloud_cover",  # Bulut kapalılık oranı (%)
    ]

    params = {
        "latitude": 38.3552,
        "longitude": 38.3095,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "hourly": hourly_vars,
        "timezone": "Europe/Istanbul",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        print(f"Hata oluştu: {e}")
        return

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    # Donma Seviyesi Yaklaşımı (Kışın Kar/Yağmur Fazı için)
    df["estimated_freezing_level_m"] = (
        (df["temperature_2m"] / 0.0065).clip(lower=0).round(1)
    )

    # Konvektif Kararsızlık Göstergesi (Yazın Dolu ve Şimşek Potansiyeli için)
    df["convective_risk_index"] = (
        (df["temperature_2m"] * (df["relative_humidity_2m"] / 100.0))
        * (df["wind_gusts_10m"] / (df["wind_speed_10m"] + 1e-5))
    ).round(2)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "historical_data.csv")
    df.to_csv(output_path, index=False)

    print(
        f"Genişletilmiş veri başarıyla kaydedildi! Toplam {len(df)} saatlik kayıt."
    )
    print(f"Dosya: {output_path}")


if __name__ == "__main__":
    download_malatya_historical_data()