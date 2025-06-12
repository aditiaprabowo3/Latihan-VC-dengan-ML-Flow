import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Set tracking URI ke /tmp/mlruns (aman untuk CI/CD)
    mlflow.set_tracking_uri("file:///tmp/mlruns")

    # Ambil dataset path
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    )
    data = pd.read_csv(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )

    # Ambil param dari CLI args
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    input_example = X_train.iloc[0:5]  # pakai .iloc buat aman

    # MLflow run
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Prediksi dan log model
        predicted = model.predict(X_test)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Log param dan metric
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
# Mungkin tebersit di benak Anda sebuah pertanyaan “Mengapa kita menghapus tracking_uri? Bukankah dengan begitu kita tidak bisa menyimpan artefak dan metrics yang dihasilkan?” Jika Anda berpikir demikian, jawabannya adalah benar teman-teman. Namun, tracking_uri ini dapat digunakan kembali ketika Anda memiliki sebuah VM atau server yang terhubung dengan MLflow Tracking UI.  

# Alasan latihan ini menghilangkan tracking_uri karena kita tidak akan menggunakan atau membuat VM/server (setidaknya sampai pada materi ini) untuk menyimpan seluruh metrics beserta artefak model yang dibuat. Selain itu, GitHub Actions sebagai platform untuk menjalankan CI pada proyek ini tidak dapat mengakses ip local atau remote repository MLflow Tracking UI. 
# Lalu, bagaimana solusinya? Salah satu yang bisa dilakukan ketika tidak memiliki bucket, VM, Server, dan lain sebagainya yaitu menyimpan seluruh logging model pada repositori GitHub itu sendiri. 

# Tentunya dengan menggunakan cara ini, Anda tidak dapat melihat UI dan membandingkan model secara langsung, tetapi hal itu bukan menjadi masalah besar. Asumsinya, penggunaan MLproject ini sudah diatur sedemikian rupa sehingga ketika dijalankan akan menghasilkan model terbaik yang sudah Anda hasilkan sebelumnya.


# Sejatinya, best practices untuk menjalankan MLflow Project adalah ketika Anda sudah memiliki tim yang bekerja bersamaan beserta infrastruktur yang mendukung. Nantinya, seluruh perubahan yang dibuat oleh ML Engineer akan terintegrasi dengan sebuah VM yang dapat dimonitoring oleh Lead atau stakeholder yang bertanggung jawab untuk melakukan deployment.

# Pro Tips
# Sejatinya, best practices untuk menjalankan MLflow Project adalah ketika Anda sudah memiliki tim yang bekerja bersamaan beserta infrastruktur yang mendukung. Nantinya, seluruh perubahan yang dibuat oleh ML Engineer akan terintegrasi dengan sebuah VM yang dapat dimonitoring oleh Lead atau stakeholder yang bertanggung jawab untuk melakukan deployment.

# Kondisi ini juga bisa dilakukan dengan catatan Anda bersedia untuk membayar sebuah VM atau server untuk menjalankan MLflow Tracking UI secara online.
# Sebenarnya, terdapat satu buah server gratis untuk menggunakan tracking secara online yaitu Dagshub tetapi dengan syarat dan ketentuan yang berlaku.