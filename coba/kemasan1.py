import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

# Path ke gambar sampel
sample_image_path = 'test.jpg'  # Ganti dengan path ke gambar sampel Anda

# Membaca gambar sampel
sample_image = cv2.imread(sample_image_path)

# Pastikan gambar berhasil dibaca
if sample_image is None:
    print("Gagal membaca gambar. Periksa path file.")
else:
    # Lakukan pemrosesan citra untuk mendeteksi blitzer
    sample_hsv = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)

    # Definisikan rentang warna untuk putih dan abu-abu
    lower_gray = np.array([0, 0, 200])
    upper_gray = np.array([180, 25, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 255, 255])

    # Mencari area berwarna abu-abu dan putih dalam gambar sampel
    mask_gray = cv2.inRange(sample_hsv, lower_gray, upper_gray)
    mask_white = cv2.inRange(sample_hsv, lower_white, upper_white)

    # Menggabungkan mask abu-abu dan putih
    mask = cv2.bitwise_or(mask_gray, mask_white)

    # Menerapkan mask ke gambar sampel
    result = cv2.bitwise_and(sample_image, sample_image, mask=mask)

    # Menemukan kontur pada mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur berdasarkan area untuk menghindari noise dan menghitung jumlah blitzer yang terdeteksi
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Menghitung jumlah blitzer berdasarkan kontur yang valid
    blitzer_count = len(valid_contours)

    # Menggambar kotak di sekeliling blitzer yang terdeteksi
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menampilkan gambar hasil deteksi
    cv2.imshow('Hasil Deteksi Blitzer', result)

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

    cv2.waitKey(0)
    cv2.destroyAllWindows()
