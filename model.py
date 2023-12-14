import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import data
data = pd.read_csv("data_UTS.csv")
data = data.drop(data.columns[0:3], axis=1)
print(data.head())
print(data.columns)

# Pilih kolom-kolom yang ingin Anda jadikan input (variabel independen)
input_columns = data[['freq_wasting', 'freq_Postpone', 'freq_pastDeadline', 'duration_of_use']]

# Menggunakan metode elbow untuk menentukan jumlah klaster yang optimal
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(input_columns)
    inertia.append(kmeans.inertia_)

# Plot metode elbow
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
plt.xlabel('Jumlah Klaster')
plt.ylabel('Inersia (WCSS)')
plt.show()

# Langkah klustering dengan K-Means
n_clusters = 2  # Ganti dengan jumlah cluster yang diinginkan
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(input_columns)

# Gantilah nilai output dengan label klaster
output_column = data['cluster']

# membuat dataframe hasil cluster
output_column.to_csv('cluster_labels.csv', index=False)
data2 = pd.read_csv("cluster_labels.csv")

dataFix = pd.concat([data, data2['cluster']],  axis=1)
dataFix.to_csv('hasil_gabungan.csv', index=False)

# Visualisasi klaster (Scatter plot untuk setiap pasangan fitur)
sns.set(style="ticks")
sns.pairplot(data, hue="cluster", palette="viridis")
plt.show()

# Memisahkan data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(input_columns, output_column, test_size=0.2, random_state=42)

# Metode: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluasi hasil klasifikasi
rf_pred = rf_model.predict(X_test)

print("Random Forest:")
print(f'Accuracy: {accuracy_score(y_test, rf_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}')
print(f'Classification Report:\n{classification_report(y_test, rf_pred)}')

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree:")
print(f'Accuracy: {accuracy_score(y_test, dt_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, dt_pred)}')
print(f'Classification Report:\n{classification_report(y_test, dt_pred)}')

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("Naive Bayes:")
print(f'Accuracy: {accuracy_score(y_test, nb_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, nb_pred)}')
print(f'Classification Report:\n{classification_report(y_test, nb_pred)}')

# Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("Support Vector Machine (SVM):")
print(f'Accuracy: {accuracy_score(y_test, svm_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, svm_pred)}')
print(f'Classification Report:\n{classification_report(y_test, svm_pred)}')
