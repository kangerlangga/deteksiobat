import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

# Path ke gambar sampel
sample_image_path = 'test3.jpeg'  # Ganti dengan path ke gambar sampel Anda

# Membaca gambar sampel
sample_image = cv2.imread(sample_image_path)

# Pastikan gambar berhasil dibaca
if sample_image is None:
    print("Gagal membaca gambar. Periksa path file.")
else:
    # Lakukan pemrosesan citra, misalnya deteksi warna kapsul
    sample_hsv = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)

    # Definisikan rentang warna untuk merah dan oranye
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Mencari area berwarna merah dalam gambar sampel
    mask_red = cv2.inRange(sample_hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(sample_hsv, lower_orange, upper_orange)

    # Menggabungkan mask merah dan oranye
    mask = cv2.bitwise_or(mask_red, mask_orange)

    # Menerapkan mask ke gambar sampel
    result = cv2.bitwise_and(sample_image, sample_image, mask=mask)

    # Menemukan kontur pada mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur berdasarkan area untuk menghindari noise dan menghitung jumlah kapsul yang terdeteksi
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Menghitung jumlah kapsul berdasarkan kontur yang valid
    capsule_count = 0
    for contour in valid_contours:
        # Menggambar kotak di sekeliling kapsul yang terdeteksi
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        capsule_count += 1  # Tambah jumlah kapsul

    # Menampilkan gambar hasil deteksi
    cv2.imshow('Hasil Deteksi Kapsul', result)

    # Menampilkan jumlah kapsul terdeteksi
    print(f'Jumlah kapsul terdeteksi: {capsule_count}')

    # Mengubah jumlah kapsul ke dalam suara berdasarkan jumlah kapsul
    if capsule_count == 12:
        text_to_speak = "Sempurna."  # Pesan jika jumlah kapsul 12
    elif capsule_count < 12:
        text_to_speak = f'Kurang {12 - capsule_count}.'  # Pesan jika kurang dari 12
    else:
        text_to_speak = f'Lebih {capsule_count - 12}.'  # Pesan jika lebih dari 12

    tts = gTTS(text=text_to_speak, lang='id')  # Menggunakan bahasa Indonesia
    audio_file = "capsule_count.mp3"
    tts.save(audio_file)

    # Memutar suara
    playsound(audio_file)

    # Menghapus file audio setelah diputar
    os.remove(audio_file)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
