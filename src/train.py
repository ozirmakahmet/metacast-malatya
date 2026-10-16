import json
import os
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -25, 25)))


def train_metacast_model(data_path: str, model_output_path: str):
    print("Genişletilmiş şiddetli hava veri seti yükleniyor...")
    df = pd.read_csv(data_path)

    # Modele giren fiziksel öznitelikler (Features)
    features = [
        "wind_speed_10m",
        "pressure_diff_3h",
        "mountain_blockage_index",
        "gust_factor",
        "is_snow_phase",
        "hail_potential",
    ]

    # Hedef Değişken: Genel Şiddetli Hava Olayı (Fırtına, Kar veya Dolu)
    target_col = "target_severe_event"

    X = df[features].values
    y = df[target_col].values

    # Dengeli rastgele bölme (%80 Eğitim, %20 Test)
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    split_idx = int(len(df) * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Z-Score Standardizasyonu
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-7

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    severe_events_train = int(np.sum(y_train))
    severe_events_test = int(np.sum(y_test))

    print(
        f"Eğitim Kümesi: {len(X_train)} saat (Şiddetli Olay: {severe_events_train}) | "
        f"Test Kümesi: {len(X_test)} saat (Şiddetli Olay: {severe_events_test})"
    )
    print("Çok tehlikeli (Multi-Hazard) karar motoru eğitiliyor...")

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
    print(
        f"Yakalanan Şiddetli Hava Olayları (Recall): %{recall * 100:.1f}"
    )
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