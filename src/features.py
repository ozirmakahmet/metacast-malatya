import os
import numpy as np
import pandas as pd


def compute_features(input_path: str, output_path: str):
    """Malatya genişletilmiş hava verilerini işler;

    orografi, basınç eğrisi, kar/yağmur faz ayrımı ve konvektif riskleri
    hesaplar.
    """
    print("Genişletilmiş fiziksel öznitelikler hesaplanıyor...")
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 1. Rüzgâr & Basınç Fiziği
    # 3 saatlik barometre eğilimi
    df["pressure_diff_3h"] = df["surface_pressure"].diff(periods=3).fillna(0)

    # Beydağları blokaj katsayısı (215 derece Güneybatı ekseni)
    mountain_angle = 215.0
    angle_diff = np.radians(df["wind_direction_10m"] - mountain_angle)
    df["mountain_blockage_index"] = np.abs(np.cos(angle_diff))

    # Hamle / Ortalama Hız Çarpanı
    df["gust_factor"] = (
        df["wind_gusts_10m"] / (df["wind_speed_10m"] + 1e-5)
    ).clip(1.0, 5.0)

    # 2. Kar / Yağmur Faz Ayrımı Fiziği
    # Malatya havzasında yer sıcaklığı 1.5°C altında ve yağış varsa kar/sulu kar fazına geçer
    df["is_snow_phase"] = (
        (df["temperature_2m"] <= 1.5) & (df["precipitation"] > 0)
    ).astype(int)

    # 3. Dolu ve Konvektif Fırtına Göstergesi
    # Yüksek konvektif indeks + ani yağış + şiddetli hamle kesişimi
    df["hail_potential"] = (
        (df["convective_risk_index"] >= 15.0)
        & (df["precipitation"] >= 1.0)
        & (df["wind_gusts_10m"] >= 35.0)
    ).astype(int)

    # 4. Hedef Değişkenler (Labels)
    # Şiddetli Rüzgâr / Fırtına (>= 40 km/s)
    df["target_storm"] = (df["wind_gusts_10m"] >= 40.0).astype(int)

    # Genel Şiddetli Hava Olayı (Fırtına VEYA Yoğun Kar VEYA Dolu/Konveksiyon)
    df["target_severe_event"] = (
        (df["target_storm"] == 1)
        | (df["snowfall"] >= 0.5)
        | (df["hail_potential"] == 1)
    ).astype(int)

    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Başarılı! Genişletilmiş özellik tablosu kaydedildi: {output_path}")
    print(f"Toplam Saat: {len(df)}")
    print(f"Tespit Edilen Fırtına Saati: {df['target_storm'].sum()}")
    print(
        f"Tespit Edilen Kar/Donma Fazı Saati: {df['is_snow_phase'].sum()}"
    )
    print(
        f"Tespit Edilen Dolu/Konvektif Risk Saati: {df['hail_potential'].sum()}"
    )


if __name__ == "__main__":
    input_file = os.path.join("data", "historical_data.csv")
    output_file = os.path.join("data", "features_data.csv")
    compute_features(input_file, output_file)