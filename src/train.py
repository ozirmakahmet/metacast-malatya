import json
import os
import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -25, 25)))

def train_metacast_model(data_path: str, model_output_path: str):
    print("Genişletilmiş şiddetli hava veri seti yükleniyor...")
    df = pd.read_csv(data_path)

    # Malatya'nın fiziksel ve topoğrafik gerçeklerine göre güncellenmiş özellikler
    features = [
        "wind_u",                 # Rüzgar Doğu-Batı Vektörü
        "wind_v",                 # Rüzgar Kuzey-Güney Vektörü
        "pressure_diff_3h",       # Barometrik Eğilim
        "temp_diff_3h",           # Çanak İçi Soğuma Hızı (Katabatik İnversiyon için)
        "mountain_blockage_index",# Beydağları Blokajı
        "gust_factor",            # Hamle Çarpanı
        "dam_moisture_advection", # Karakaya Barajı Nem Taşınımı
        "is_snow_phase",          # Kar/Sulu Kar Fazı
        "hail_potential",         # Konvektif Dolu Potansiyeli
    ]

    # Hedef Değişken: Genel Şiddetli Hava Olayı (Fırtına, Kar, Dolu VEYA Zirai Don)
    target_col = "target_severe_event"

    X = df[features].values
    y = df[target_col].values

    # ZAMANSAL BÖLME (Temporal Split) - Data Leakage Engellendi
    # Hava durumu verisi rastgele karıştırılamaz, kronolojik ilerlemelidir.
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
        f"Eğitim Kümesi (Geçmiş): {len(X_train)} saat (Şiddetli Olay: {severe_events_train}) | "
        f"Test Kümesi (Gelecek): {len(X_test)} saat (Şiddetli Olay: {severe_events_test})"
    )
    print("Fizik kısıtlı (Physics-Informed) karar motoru eğitiliyor...")

    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    lr = 0.05
    # Sınıf dengesizliği (Class Imbalance) için pozitif sınıf ağırlığı
    pos_weight = (len(y_train) - np.sum(y_train)) / (np.sum(y_train) + 1e-5)

    for epoch in range(1500):
        linear_model = np.dot(X_train_norm, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Hata hesaplamasında sınıf ağırlığı uygulanıyor
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
    print(f"Karmaşıklık: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model_payload = {
        "features": features,
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    with open(model_output_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\nModel başarıyla kaydedildi: {model_output_path}")

if __name__ == "__main__":
    input_features = os.path.join("data", "features_data.csv")
    output_model = os.path.join("models", "metacast_weights.json")
    train_metacast_model(input_features, output_model)
