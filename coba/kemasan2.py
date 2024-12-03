import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

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
        
        # Lakukan pemrosesan citra untuk mendeteksi blitzer
        sample_hsv = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)

        # Memperketat rentang warna untuk abu-abu dan putih
        lower_gray = np.array([0, 0, 180])  # Rentang abu-abu ketat
        upper_gray = np.array([180, 30, 230])
        lower_white = np.array([0, 0, 220])  # Rentang putih ketat
        upper_white = np.array([180, 25, 255])

        # Mencari area berwarna abu-abu dan putih dalam gambar sampel
        mask_gray = cv2.inRange(sample_hsv, lower_gray, upper_gray)
        mask_white = cv2.inRange(sample_hsv, lower_white, upper_white)

        # Menggabungkan mask abu-abu dan putih
        mask = cv2.bitwise_or(mask_gray, mask_white)

        # Menemukan kontur pada mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter kontur berdasarkan area untuk menghindari noise dan menghitung jumlah blitzer yang terdeteksi
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]  # Sesuaikan threshold area

        # Menghitung jumlah blitzer berdasarkan kontur yang valid
        blitzer_count = len(valid_contours)

        # Menggambar kotak di sekeliling blitzer yang terdeteksi pada gambar asli
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Menampilkan gambar asli dengan kotak di sekitar blitzer yang terdeteksi
        cv2.imshow('Hasil Deteksi Blitzer (Gambar Asli)', sample_image)

        # Menampilkan jumlah blitzer terdeteksi
        print(f'Jumlah blitzer terdeteksi: {blitzer_count}')

        # Mengubah jumlah blitzer ke dalam suara
        text_to_speak = f'Ada {blitzer_count} blitzer'
        tts = gTTS(text=text_to_speak, lang='id')  # Menggunakan bahasa Indonesia
        audio_file = "blitzer_count.mp3"
        tts.save(audio_file)

        # Memutar suara
        playsound(audio_file)

        # Menghapus file audio setelah diputar
        os.remove(audio_file)

    elif key == ord('q'):
        # Keluar dari loop jika tombol 'q' ditekan
        break

# Membersihkan resource
cap.release()
cv2.destroyAllWindows()
