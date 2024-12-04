import cv2
import os
from datetime import datetime

# Membuat folder 'capture' jika belum ada
output_folder = "capture"
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi kamera (webcam)
cap = cv2.VideoCapture(0)  # 0 untuk kamera default

if not cap.isOpened():
    print("Tidak dapat mengakses kamera")
    exit()

print("Tekan 'c' untuk mengambil gambar, atau 'q' untuk keluar.")

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal membaca frame dari kamera")
        break
    
    # Tampilkan frame di jendela
    cv2.imshow('Webcam', frame)
    
    # Deteksi input keyboard
    key = cv2.waitKey(1) & 0xFF  # Tunggu 1ms untuk input
    if key == ord('c'):
        # Buat nama file unik berdasarkan waktu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'captured_image_{timestamp}.png'
        filepath = os.path.join(output_folder, filename)  # Gabungkan path folder dan nama file
        cv2.imwrite(filepath, frame)
        print(f"Gambar disimpan di {filepath}")
    elif key == ord('q'):
        # Keluar saat tombol 'q' ditekan
        print("Keluar dari program.")
        break

# Melepaskan kamera dan menutup semua jendela
cap.release()
cv2.destroyAllWindows()
