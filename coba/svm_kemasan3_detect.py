import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Fungsi untuk mengekstrak fitur HOG
def extract_hog_features(image):
    # Ubah ukuran gambar menjadi 128x128 agar sesuai dengan data pelatihan
    image_resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(16, 16), 
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

# Memuat model SVM yang sudah dilatih
model = joblib.load("blitzer_one_class_model.pkl")

# Membuka akses ke webcam
cap = cv2.VideoCapture(0)

print("Tekan 'c' untuk mengambil gambar dan mendeteksi blitzer. Tekan 'q' untuk keluar.")

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca dari webcam.")
        break

    # Tampilkan frame dari webcam
    cv2.imshow("Webcam", frame)

    # Tunggu input tombol
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Menyimpan gambar yang diambil
        sample_image = frame.copy()
        
        # Ekstrak fitur HOG dari gambar yang diambil
        features = extract_hog_features(sample_image).reshape(1, -1)

        # Prediksi menggunakan model One-Class SVM
        prediction = model.predict(features)

        # Jika blister pack terdeteksi
        if prediction[0] == 1:
            cv2.putText(sample_image, "Blister Terdeteksi", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(sample_image, (10, 10), (138, 138), (0, 255, 0), 2)
        else:
            cv2.putText(sample_image, "Blister Tidak Terdeteksi", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Menampilkan gambar hasil deteksi
        cv2.imshow('Hasil Deteksi Blister', sample_image)

    elif key == ord('q'):
        # Keluar dari loop jika tombol 'q' ditekan
        break

# Membersihkan resource
cap.release()
cv2.destroyAllWindows()
