import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(npz_path: str = "ozellikler.npz"):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    ages = data["age"]
    return X, ages


def age_to_bucket(age_value):
    """
    Yaşı aralığa çevirir.
    Common Voice etiketleri için örnekler:
      teens, twenties, thirties, fourties, fifties, sixties, seventies, eighties, nineties

    Gruplar:
      0-19  -> '0-19'
      20-29 -> '20-29'
      30-39 -> '30-39'
      40+   -> '40+'
    """
    if age_value is None:
        return None

    s = str(age_value).strip().lower()
    if s == "":
        return None

    # Önce sayı varsa direkt kullan
    try:
        age_num = int(s)
        if age_num < 20:
            return "0-19"
        elif age_num < 30:
            return "20-29"
        elif age_num < 40:
            return "30-39"
        else:
            return "40+"
    except ValueError:
        pass

    # String yaş gruplarını mapping ile dönüştür
    mapping = {
        "teens": "0-19",
        "twenties": "20-29",
        "thirties": "30-39",
        "fourties": "40+",
        "forties": "40+",
        "fifties": "40+",
        "sixties": "40+",
        "seventies": "40+",
        "eighties": "40+",
        "nineties": "40+",
    }

    return mapping.get(s)


def prepare_data(X, ages):
    # Yaşları yaş aralığı etiketine çevir
    buckets = [age_to_bucket(a) for a in ages]

    # Geçersiz (None) olanları filtrele
    valid_indices = [i for i, b in enumerate(buckets) if b is not None]

    if len(valid_indices) == 0:
        raise ValueError(
            "Hiç geçerli yaş etiketi bulunamadı. 'age' sütunundaki örneklerden ilk birkaçını kontrol edin."
        )

    X_valid = X[valid_indices]
    buckets_valid = np.array([buckets[i] for i in valid_indices])

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid,
        buckets_valid,
        test_size=0.2,
        random_state=42,
        stratify=buckets_valid,
    )

    return X_train, X_test, y_train, y_test


def build_model():
    # Standartlaştırma + RandomForest pipeline'ı
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )
    return model


def train_and_evaluate():
    print("Veri yükleniyor (yaş aralıkları için)...")
    X, ages = load_data("ozellikler.npz")

    print(f"Toplam örnek sayısı (ham): {len(X)}")

    X_train, X_test, y_train, y_test = prepare_data(X, ages)

    print(f"Geçerli yaş etiketi olan örnek sayısı (filtrelenmiş): {len(X_train) + len(X_test)}")

    print("Yaş aralığı modeli eğitiliyor...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Test seti üzerinde değerlendirme yapılıyor...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nYaş aralığı tahmini doğruluğu (accuracy): {acc:.4f}\n")

    print("Sınıflandırma raporu:")
    print(classification_report(y_test, y_pred))

    print("Karmaşıklık matrisi (confusion matrix):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate()

