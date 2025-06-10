import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Credit Scoring")

data = pd.read_csv("train_pca.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]
# Mendifinisikan model menggunakan hyperparameter tuning.
# Define Elastic Search parameters
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int)  # 5 evenly spaced values

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            mlflow.autolog()

            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                    )
                
# Kode di atas mendemonstrasikan proses hyperparameter tuning menggunakan model RandomForestClassifier dengan bantuan MLflow untuk mencatat eksperimen.
# Hyperparameter yang diuji adalah jumlah estimator (n_estimators) dan kedalaman maksimum pohon keputusan (max_depth). 

# Kedua parameter ini diatur untuk dieksplorasi dalam rentang nilai tertentu menggunakan metode perulangan yang setiap kombinasi nilainya diuji untuk menemukan model dengan akurasi terbaik. Detailnya, kode di atas akan menjalankan langkah-langkah seperti berikut.

'''
1. Menentukan Rentang Hyperparameter
Pertama, kita perlu mendefinisikan rentang nilai untuk dua hyperparameter yang akan digunakan.
- n_estimators_range: menggunakan fungsi np.linspace untuk menghasilkan lima nilai secara acak di antara 10 sampai 1000.
- max_depth_range: menghasilkan lima nilai yang terdistribusi merata antara 1 sampai 50.

2. Variabel untuk Menyimpan Hasil Terbaik
kita perlu mendefinisikan dua variabel yang akan digunakan untuk melacak hasil terbaik selama iterasi dijalankan.
- best_accuracy: menyimpan akurasi tertinggi yang dicapai selama proses pelatihan.
- best_params: menyimpan kombinasi hyperparameter yang menghasilkan akurasi terbaik.

3. Loop Untuk Hyperparameter Tuning
Perulangan ini ditandai dengan dua buah loop yang bergandengan dengan tugas yang berbeda.
- Loop pertama bertugas untuk mengulang seluruh data berdasarkan nilai n_estimators.
- Loop kedua bertugas untuk mengulang seluruh data berdasarkan nilai max_depth.

4. Mencatat Eksperimen Menggunakan MLflow
Setiap kombinasi parameter diuji dalam eksperimen MLflow yang terpisah menggunakan with mlflow.start_run(). Nama eksperimen diberikan berdasarkan kombinasi hyperparameter saat ini (elastic_search_<n_estimators>_<max_depth>).

5. Training dan Evaluation Model
- Model RandomForestClassifier dibuat berdasarkan nilai hyperparameter n_estimators dan max_depth. Di lain sisi, kita juga menggunakan hyperparameter random_state=42 untuk hasil yang konsisten.
- Model dilatih menggunakan data pelatihan (X_train, y_train) dengan metode .fit().
- Akurasi model dihitung menggunakan data pengujian (X_test, y_test) melalui metode .score().

6. Mencatat Seluruh Metrics pada MLflow Tracking
Dengan mlflow.autolog(), semua parameter, metrik, dan artefak model secara otomatis dicatat ke MLflow Tracking. Di lain sisi, kode ini juga menambahkan akurasi model sebagai metrics tambahan secara eksplisit menggunakan mlflow.log_metric("accuracy", accuracy) agar dapat dibandingkan secara langsung.

7. Menyimpan Model Yang Lebih Baik
Jika akurasi model yang dibangun lebih tinggi daripada best_accuracy sebelumnya, hasil tersebut akan memperbarui variabel best_accuracy dan menyimpan model pada Tracking.
'''

# Apakah sampai di sini semuanya sudah selesai? Tentunya tidak, karena selanjutnya kita perlu membuat MLproject agar dapat melakukan proses CI menggunakan GitHub Actions. Hal pertama yang perlu kita lakukan yaitu membuat sebuah folder MLproject yang berisikan dependensi beserta perintah untuk menjalankan MLflow run. 


