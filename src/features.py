import os
import numpy as np
import pandas as pd

def compute_features(input_path: str, output_path: str):
    print("Genişletilmiş fiziksel öznitelikler hesaplanıyor...")
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 1. Rüzgâr Fiziği ve Trigonometrik Dönüşüm (Kritik ML Optimizasyonu)
    # Rüzgar yönünü U (Doğu-Batı) ve V (Kuzey-Güney) vektörlerine ayırıyoruz.
    wd_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(wd_rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(wd_rad)

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

    # 2. Karakaya Barajı Termal Atalet ve Nem Etkisi
    # Rüzgar Doğu sektöründen (45-135 derece) estiğinde nem taşınımı artar.
    is_east_wind = (df["wind_direction_10m"] >= 45) & (df["wind_direction_10m"] <= 135)
    df["dam_moisture_advection"] = is_east_wind.astype(int) * (df["relative_humidity_2m"] / 100.0) * df["wind_speed_10m"]

    # Çanak Etkisi ve Termal İnversiyon (Katabatik Soğuma Hızı)
    df["temp_diff_3h"] = df["temperature_2m"].diff(periods=3).fillna(0)

    # 3. Kar / Yağmur Faz Ayrımı Fiziği
    df["is_snow_phase"] = (
        (df["temperature_2m"] <= 1.5) & (df["precipitation"] > 0)
    ).astype(int)

    # 4. Dolu ve Konvektif Fırtına Göstergesi
    df["hail_potential"] = (
        (df["convective_risk_index"] >= 15.0)
        & (df["precipitation"] >= 1.0)
        & (df["wind_gusts_10m"] >= 35.0)
    ).astype(int)

    # 5. Hedef Değişkenler (Labels)
    # Şiddetli Rüzgâr / Fırtına (>= 40 km/s)
    df["target_storm"] = (df["wind_gusts_10m"] >= 40.0).astype(int)

    # Zirai Don Riski (Malatya Tarımı İçin Kritik Hedef)
    # Sıcaklık 0'ın altında, bulut az, rüzgar sakinse (radyasyon donu)
    df["target_frost"] = (
        (df["temperature_2m"] <= 0.0) 
        & (df["cloud_cover"] <= 30.0) 
        & (df["wind_speed_10m"] <= 5.0)
    ).astype(int)

    # Genel Şiddetli Hava Olayı
    df["target_severe_event"] = (
        (df["target_storm"] == 1)
        | (df["snowfall"] >= 0.5)
        | (df["hail_potential"] == 1)
        | (df["target_frost"] == 1)
    ).astype(int)

    df = df.dropna().reset_index(drop=True)
    
    # Modelin kafasını karıştırmamak için döngüsel ham rüzgar yönünü çıkarıyoruz
    df = df.drop(columns=["wind_direction_10m"])
    
    df.to_csv(output_path, index=False)

    print(f"Başarılı! Genişletilmiş özellik tablosu kaydedildi: {output_path}")
    print(f"Toplam Saat: {len(df)}")
    print(f"Tespit Edilen Fırtına Saati: {df['target_storm'].sum()}")
    print(f"Tespit Edilen Zirai Don Saati: {df['target_frost'].sum()}")

if __name__ == "__main__":
    input_file = os.path.join("data", "historical_data.csv")
    output_file = os.path.join("data", "features_data.csv")
    compute_features(input_file, output_file)
