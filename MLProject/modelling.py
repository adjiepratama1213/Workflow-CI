import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # 1. Konfigurasi MLflow Tracking UI (Local)
    # mlflow.set_tracking_uri("http://127.0.0.1:5000") # Di-comment agar bisa jalan di GitHub Actions (simpan di mlruns)
    # mlflow.set_experiment("Titanic_Basic_Model") # Di-comment agar tidak bentrok dengan id mlflow run

    # 2. Aktifkan autologging untuk scikit-learn
    mlflow.sklearn.autolog()

    # 3. Memuat data preprocessed
    df = pd.read_csv('titanic_preprocessing/titanic_preprocessed.csv')

    # 4. Memisahkan fitur (X) dan target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 5. Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # 6. Membangun dan melatih model dasar
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # 7. Melakukan prediksi pada data testing
        y_pred = model.predict(X_test)

        # 8. Menghitung akurasi
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Catat metrik secara manual (opsional, karena autologging sudah mencatatnya)
        mlflow.log_metric("accuracy", accuracy)

        # 9. Menyimpan (log) model
        mlflow.sklearn.log_model(model, "model")

        print("Model berhasil dilatih dan disimpan di MLflow.")

if __name__ == "__main__":
    main()
