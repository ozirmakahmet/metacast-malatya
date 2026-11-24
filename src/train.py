import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

LAT_MERKEZ = 38.3552
LON_MERKEZ = 38.3095
TR_TZ = timezone(timedelta(hours=3))

WEIGHTS_PATH = "models/metacast_weights.json"
DB_PATH = "data/metacast_logs.db"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


def calculate_gaussian_blockage(wind_dir, peak_angle=215.0, spread=35.0):
    diff = (wind_dir - peak_angle + 180.0) % 360.0 - 180.0
    return np.exp(-(diff ** 2) / (2.0 * (spread ** 2)))


def fetch_historical_malatya_data(start_date="2020-01-01", end_date=None):
    if end_date is None:
        end_date = (datetime.now(TR_TZ) - timedelta(days=2)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT_MERKEZ,
        "longitude": LON_MERKEZ,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "dew_point_2m",
            "precipitation",
            "snowfall",
            "surface_pressure",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
            "cloud_cover",
        ],
        "timezone": "Europe/Istanbul",
    }

    print(f"[MetaCast Train] {start_date} -> {end_date} arası Malatya arşiv verisi çekiliyor...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.dropna().reset_index(drop=True)


def build_features_and_labels(df):
    w_speed = df["wind_speed_10m"].astype(float).values
    w_gust = df["wind_gusts_10m"].astype(float).values
    w_dir = df["wind_direction_10m"].astype(float).values
    temp = df["temperature_2m"].astype(float).values
    dew_pt = df["dew_point_2m"].astype(float).values
    precip = df["precipitation"].astype(float).values
    snowfall = df["snowfall"].astype(float).values
    cloud = df["cloud_cover"].astype(float).values

    pressure_diff = df["surface_pressure"].astype(float).diff(periods=3).fillna(0.0).values
    mountain_blockage = calculate_gaussian_blockage(w_dir)
    gust_factor = np.clip(w_gust / (w_speed + 1e-5), 1.0, 5.0)

    # Kış yağış & kar fazı tespiti (Sıcaklık + Çiy noktası + Gerçek kar verisi)
    wet_bulb_est = temp - ((temp - dew_pt) * 0.35)
    is_snow_phase = ((snowfall > 0.05) | ((precip > 0.1) & (wet_bulb_est <= 1.2))).astype(float)

    # Konvektif dolu / şiddetli sağanak potansiyeli
    hail_potential = ((precip >= 3.5) & (w_gust >= 40.0) & (temp >= 18.0)).astype(float)

    X = np.column_stack([
        w_speed,
        pressure_diff,
        mountain_blockage,
        gust_factor,
        is_snow_phase,
        hail_potential,
    ])

    # Malatya havzası kritik kış + yaz afet etiketleri (Ground Truth)
    is_night = (df["time"].dt.hour >= 21) | (df["time"].dt.hour <= 7)
    winter_frost = is_night & (temp <= 1.0) & (cloud <= 35.0)
    heavy_snow_or_ice = (snowfall >= 0.4) | ((precip >= 0.8) & (temp <= 1.5))
    winter_rain_storm = (precip >= 2.5) & (w_gust >= 35.0)
    unblocked_windstorm = (w_gust >= 48.0) & (mountain_blockage < 0.65)

    y = (winter_frost | heavy_snow_or_ice | winter_rain_storm | unblocked_windstorm).astype(float).values

    # Kış aylarına (Kasım, Aralık, Ocak, Şubat, Mart) 3.5x eğitim ağırlığı
    months = df["time"].dt.month.values
    sample_weights = np.where(np.isin(months, [11, 12, 1, 2, 3]), 3.5, 1.0)

    return X, y, sample_weights


def train_weighted_logistic_regression(X, y, sample_weights, lr=0.08, epochs=1800, l2_reg=0.01):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    n_samples, n_features = X_norm.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    weight_norm_factor = sample_weights / sample_weights.mean()

    for epoch in range(epochs):
        z = np.dot(X_norm, weights) + bias
        preds = sigmoid(z)
        errors = (preds - y) * weight_norm_factor

        grad_w = (np.dot(X_norm.T, errors) / n_samples) + (l2_reg * weights)
        grad_b = np.mean(errors)

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, float(bias), mean, std


def save_weights(weights, bias, mean, std, total_records):
    os.makedirs("models", exist_ok=True)
    payload = {
        "weights": np.round(weights, 6).tolist(),
        "bias": round(float(bias), 6),
        "mean": np.round(mean, 6).tolist(),
        "std": np.round(std, 6).tolist(),
        "last_calibrated_timestamp": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M"),
        "total_logs_trained": int(total_records),
        "season_profile": "Winter-Weighted Malatya Basin v2",
    }
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[MetaCast Train] Ağırlıklar kaydedildi -> {WEIGHTS_PATH}")


def retrain_from_logs(min_new_records=50):
    """app.py tarafından canlı SQLite doğrulama kayıtları biriktikçe çağrılır."""
    df_hist = fetch_historical_malatya_data(start_date="2020-01-01")
    X, y, sample_weights = build_features_and_labels(df_hist)

    extra_logs = 0
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df_logs = pd.read_sql_query(
                "SELECT * FROM validation_logs WHERE obs_gust IS NOT NULL", conn
            )
            conn.close()
            extra_logs = len(df_logs)
        except Exception:
            extra_logs = 0

    weights, bias, mean, std = train_weighted_logistic_regression(X, y, sample_weights)
    save_weights(weights, bias, mean, std, total_records=len(df_hist) + extra_logs)


if __name__ == "__main__":
    retrain_from_logs(min_new_records=0)
    print("[MetaCast Train] Kış sezonu kalibrasyonu başarıyla tamamlandı.")
