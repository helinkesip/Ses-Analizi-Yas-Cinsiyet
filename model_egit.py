import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(npz_path: str = "ozellikler.npz"):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    genders = data["gender"]
    # Sadece cinsiyet etiketini kullanıyoruz
    return X, genders


def prepare_data(X, genders):
    # Cinsiyet etiketlerini (ör: 'male', 'female') sayısal forma çevir
    le_gender = LabelEncoder()
    y = le_gender.fit_transform(genders)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, le_gender


def build_model():
    # Standartlaştırma + Lojistik Regresyon pipeline'ı
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return model


def train_and_evaluate():
    print("Veri yükleniyor...")
    X, genders = load_data("ozellikler.npz")

    print(f"Toplam örnek sayısı: {len(X)}")

    X_train, X_test, y_train, y_test, le_gender = prepare_data(X, genders)

    print("Model eğitiliyor...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Test seti üzerinde değerlendirme yapılıyor...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nCinsiyet tahmini doğruluğu (accuracy): {acc:.4f}\n")

    print("Sınıflandırma raporu:")
    target_names = le_gender.inverse_transform(sorted(set(y_test)))
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Karmaşıklık matrisi (confusion matrix):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate()

