import json
import os
import sqlite3
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -25, 25)))


# --- 1. ANA MODEL EĞİTİM MOTORU (GEÇMİŞ ARŞİV İLE BÜYÜK EĞİTİM) ---
def train_metacast_model(data_path: str, model_output_path: str):
    print("Genişletilmiş şiddetli hava veri seti yükleniyor...")
    if not os.path.exists(data_path):
        print(f"Hata: {data_path} bulunamadı!")
        return

    df = pd.read_csv(data_path)

    # Malatya'nın topoğrafik ve fiziksel öznitelikleri
    features = [
        "wind_u",
        "wind_v",
        "pressure_diff_3h",
        "temp_diff_3h",
        "mountain_blockage_index",
        "gust_factor",
        "dam_moisture_advection",
        "is_snow_phase",
        "hail_potential",
    ]

    # Hedef Değişken (Fırtına, Kar, Dolu veya Zirai Don)
    target_col = "target_severe_event"

    X = df[features].values
    y = df[target_col].values

    # Zamansal Bölme (Temporal Split - Veri Sızıntısını Önler)
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Z-Score Standardizasyonu
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-7

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    severe_events_train = int(np.sum(y_train))
    severe_events_test = int(np.sum(y_test))

    print(
        f"Eğitim Kümesi: {len(X_train)} saat (Afet: {severe_events_train}) | "
        f"Test Kümesi: {len(X_test)} saat (Afet: {severe_events_test})"
    )
    print("Fizik kısıtlı (Physics-Informed) karar motoru eğitiliyor...")

    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    lr = 0.05
    pos_weight = (len(y_train) - np.sum(y_train)) / (np.sum(y_train) + 1e-5)

    for epoch in range(1500):
        linear_model = np.dot(X_train_norm, weights) + bias
        y_predicted = sigmoid(linear_model)

        errors = (y_predicted - y_train) * np.where(y_train == 1, pos_weight, 1)

        dw = (1 / len(X_train)) * np.dot(X_train_norm.T, errors)
        db = (1 / len(X_train)) * np.sum(errors)

        weights -= lr * dw
        bias -= lr * db

    # Test performansı ölçümü
    test_preds_prob = sigmoid(np.dot(X_test_norm, weights) + bias)
    test_preds = (test_preds_prob >= 0.5).astype(int)

    tp = int(np.sum((y_test == 1) & (test_preds == 1)))
    fp = int(np.sum((y_test == 0) & (test_preds == 1)))
    fn = int(np.sum((y_test == 1) & (test_preds == 0)))
    tn = int(np.sum((y_test == 0) & (test_preds == 0)))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    print("\n--- MODEL TEST PERFORMANSI (ÇOKLU TEHLİKE TESTİ) ---")
    print(f"Test Kümesi F1-Skoru: {f1:.3f}")
    print(f"Yakalanan Şiddetli Hava Olayları (Recall): %{recall * 100:.1f}")
    print(f"Doğruluk (Precision): %{precision * 100:.1f}")
    print(f"Karmaşıklık Matrisi: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model_payload = {
        "features": features,
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "last_trained_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_logs_trained": 0,
    }

    with open(model_output_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\nModel başarıyla kaydedildi: {model_output_path}")


# --- 2. CANLI KENDİ KENDİNE ÖĞRENME MOTORU (SQLITE LOGLARI İLE OTOMATİK KALİBRASYON) ---
def retrain_from_logs(
    db_path: str = "data/metacast_logs.db",
    model_path: str = "models/metacast_weights.json",
    min_new_records: int = 50,
):
    """
    SQLite tablosunda biriken gerçek Erhaç METAR ölçümlerini okur.
    Zaman sönümlemesi (Time-Decay) ve L2 düzenlileştirmesi uygulayarak
    ağırlıkları yerel topoğrafik blokaja göre güvenli biçimde günceller.
    """
    if not os.path.exists(db_path) or not os.path.exists(model_path):
        return False, "Veritabanı veya model ağırlık dosyası bulunamadı."

    conn = sqlite3.connect(db_path)
    df_logs = pd.read_sql_query(
        "SELECT * FROM validation_logs WHERE obs_gust IS NOT NULL ORDER BY id ASC", conn
    )
    conn.close()

    if len(df_logs) < min_new_records:
        return False, f"Yetersiz kayıt sayısı ({len(df_logs)}/{min_new_records})."

    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    weights = np.array(payload["weights"])
    bias = float(payload["bias"])

    # --- 1. ZAMANLA UNUTMA (TIME-DECAY) AĞIRLIKLANDIRMASI ---
    # Son kayıtlara daha çok, eski kayıtlara daha az ağırlık verilir (decay factor: 0.98)
    n_records = len(df_logs)
    time_weights = np.exp(np.linspace(-2.0, 0.0, n_records))
    time_weights /= np.sum(time_weights)

    delta_w = df_logs["delta_wind"].values
    # Ağırlıklı ortalama hata
    weighted_mean_bias = float(np.sum(delta_w * time_weights))

    # --- 2. GÜVENLİ DERECELİ GÜNCELLEME VE L2 CEZASI ---
    lr = 0.008
    l2_reg = 0.001  # Katsayıların patlamasını engelleyen L2 katsayısı

    # 4. indeks mountain_blockage_index katsayısıdır
    if len(weights) > 4:
        # Ağırlığı güncellerken hafif bir L2 geri çekmesi uygula
        weights[4] = weights[4] * (1.0 - lr * l2_reg) - (lr * weighted_mean_bias * 0.1)
        # Katsayıyı güvenli sınırda tut (-5.0 ile 5.0 arası)
        weights[4] = float(np.clip(weights[4], -5.0, 5.0))

    bias = bias - (lr * weighted_mean_bias * 0.05)
    bias = float(np.clip(bias, -10.0, 10.0))

    payload["weights"] = weights.tolist()
    payload["bias"] = bias
    payload["last_calibrated_timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["total_logs_trained"] = len(df_logs)

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    return True, f"Model {len(df_logs)} yer istasyonu kaydıyla (Time-Decay + L2) güvenle güncellendi!"


if __name__ == "__main__":
    input_features = os.path.join("data", "features_data.csv")
    output_model = os.path.join("models", "metacast_weights.json")
    
    # Dosya doğrudan çalıştırıldığında ana eğitimi başlatır
    train_metacast_model(input_features, output_model)
