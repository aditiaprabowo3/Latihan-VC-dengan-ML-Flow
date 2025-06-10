import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

# agar eksperimen yang dijalankan dapat disimpan pada Tracking UI silakan tambahkan kode set_tracking_uri() 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# agar data hasil pelatihan dapat tersimpan pada satu pipeline silakan gunakan fungsi set_experiment()
mlflow.set_experiment("Latihan Credit Scoring")

data = pd.read_csv("train_pca.csv")
 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

# Jika Anda perhatikan kode di atas, kita melakukan hard code terhadap dataset yang akan digunakan. Hal tersebut kurang elok dan akan menghasilkan masalah jika Anda membutuhkan dataset yang berubah-ubah. Tenang saja, permasalahan tersebut dapat kita selesaikan ketika membuat MLproject dengan cara mengubah file_path yang sebelumnya statis menjadi parameter yang dinamis

# Sebelum kita masuk ke tahap utama pembangunan model, terdapat sebuah detail kecil yang sangat membantu ketika pengujian atau melakukan kolaborasi antar tim, yaitu menyimpan snippet atau sample input. Caranya sangat mudah, Anda hanya perlu memisahkan beberapa sampel data seperti berikut.
input_example = X_train[0:5]

# Mengapa hal ini penting? Bayangkan Anda diberikan model legacy (turunan) dari sebuah perusahaan dan tidak tahu menahu mengenai skema input yang digunakan. Tentunya sulit untuk menebak skemanya terlebih jika tidak ada dokumentasi sedikit pun sehingga Anda perlu mengorek metadata model dan ketika sudah dapat belum tentu rentang datanya benar

# Yang paling parah, Anda harus menebak-nebak input berdasarkan shape sebuah model. Tentunya itu sangat melelahkan ‘kan? Karena menebak-nebak skema model itu sangat sulit sehingga dengan adanya hal kecil seperti ini akan menghindari tim Anda dari tech debt di kemudian hari. 

# Setelah semuanya rampung, Anda perlu menggunakan sebuah fungsi yang bertugas untuk mengelola proyek machine learning mulai dari melatih, mencatat, hingga menyimpan model ke suatu direktori. 

with mlflow.start_run():
    # Log parameters
    n_estimators = 505
    max_depth = 37
    mlflow.autolog()
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

# Kode di atas dimulai dengan menggunakan fungsi with mlflow.start_run() yang bertugas untuk membuat sesi eksperimen baru untuk mencatat semua aktivitas terkait. Pada tahap awal, parameter model seperti n_estimators dan max_depth didefinisikan, lalu fitur autologging diaktifkan melalui mlflow.autolog(). Fitur ini memungkinkan MLflow untuk secara otomatis mencatat parameter, metrik, dan informasi lainnya tanpa memerlukan banyak kode tambahan.

# Model RandomForestClassifier kemudian dibuat menggunakan parameter yang telah didefinisikan sebelumnya, dan model tersebut dicatat sebagai artefak menggunakan mlflow.sklearn.log_model. Fungsi ini menyimpan model dalam format yang kompatibel dengan MLflow, termasuk informasi tentang input model melalui parameter input_example. 

# Setelah itu, model dilatih dengan data pelatihan (X_train dan y_train) menggunakan metode .fit(). Setelah pelatihan selesai, metrik akurasi dihitung pada data pengujian (X_test, y_test) menggunakan metode .score() dari model. Hasil metrik ini kemudian dicatat ke dalam eksperimen dengan fungsi mlflow.log_metric("accuracy", accuracy). Semua parameter, model, dan metrik yang dicatat selama eksperimen dapat diakses melalui antarmuka MLflow Tracking sehingga memungkinkan evaluasi dan reproduksi eksperimen di masa depan.
