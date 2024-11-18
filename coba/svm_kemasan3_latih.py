import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

# Fungsi untuk mengekstrak fitur HOG
def extract_hog_features(image):
    # Ubah ukuran gambar menjadi 128x128 agar sesuai dengan data pelatihan
    image_resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(16, 16), 
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

# Path ke folder dataset yang berisi gambar PNG blister
dataset_folder_path = "dataset/kemasan"

# Mengumpulkan semua fitur dari gambar
features = []

# Membaca semua gambar dalam folder
for filename in os.listdir(dataset_folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(dataset_folder_path, filename)
        image = cv2.imread(image_path)
        
        if image is not None:
            # Ekstrak fitur HOG dari gambar
            hog_features = extract_hog_features(image)
            features.append(hog_features)

# Konversi daftar fitur menjadi array NumPy
X = np.array(features)

# Pastikan fitur memiliki ukuran yang konsisten
print(f"Jumlah data pelatihan: {X.shape[0]}")  # Menampilkan jumlah gambar
print(f"Dimensi fitur setiap gambar: {X.shape[1]}")  # Menampilkan dimensi fitur

# Menggunakan One-Class SVM
from sklearn.svm import OneClassSVM

# Inisialisasi dan latih model One-Class SVM
model = OneClassSVM(gamma='auto')  # Anda bisa menyesuaikan parameter gamma
model.fit(X)

# Simpan model ke file
joblib.dump(model, "blitzer_one_class_model.pkl")
print("Model disimpan.")
