from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

app = Flask(__name__)

# Baca data
data = pd.read_csv("data_UTS.csv")
data = data.drop(data.columns[0:3], axis=1)

# Pilih kolom-kolom yang ingin Anda jadikan input (variabel independen)
input_columns = data[['freq_wasting', 'freq_Postpone', 'freq_pastDeadline', 'duration_of_use']]

# Langkah klustering dengan K-Means
n_clusters = 2  # Ganti dengan jumlah cluster yang diinginkan
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(input_columns)

# Gantilah nilai output dengan label klaster
output_column = data['cluster']

# Memisahkan data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(input_columns, output_column, test_size=0.2, random_state=42)

# Metode: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def describe_cluster(cluster_label):
    if cluster_label == 0:
        return "Deskripsi untuk Cluster 1: Pengguna dengan kecenderungan tidak terpengaruh pada semua faktor."
    elif cluster_label == 1:
        return "Deskripsi untuk Cluster 2: Pengguna dengan kecenderungan terpengaruh pada semua faktor."
    else:
        return "Deskripsi umum jika tidak cocok dengan kelompok yang diharapkan."


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Mendapatkan nilai dari input pengguna
            feature1_val = float(request.form['freq_wasting'])
            feature2_val = float(request.form['freq_Postpone'])
            feature3_val = float(request.form['freq_pastDeadline'])
            duration_val = float(request.form['duration_of_use'])

            # Melakukan klasifikasi
            cluster = kmeans.predict([[feature1_val, feature2_val, feature3_val, duration_val]])[0]
            prediction = rf_model.predict([[feature1_val, feature2_val, feature3_val, duration_val]])[0]
            
            # Mendapatkan deskripsi untuk klaster
            cluster_description = describe_cluster(cluster)

            return render_template('index.html', cluster=cluster, prediction=prediction, cluster_description=cluster_description)
        except ValueError:
            # Menangani kesalahan jika input bukan angka
            return render_template('index.html', error='Masukkan angka valid untuk semua fitur!')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
