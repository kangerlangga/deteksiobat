import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

# Fungsi untuk mendeteksi kapsul pada gambar yang diambil
def detect_capsules(image):
    # Mengonversi gambar ke HSV untuk mempermudah deteksi warna
    sample_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definisikan rentang warna untuk mendeteksi warna merah dan oranye
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    lower_orange = np.array([11, 70, 50])
    upper_orange = np.array([25, 255, 255])

    # Membuat mask untuk warna merah dan oranye
    mask_red = cv2.inRange(sample_hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(sample_hsv, lower_orange, upper_orange)

    # Menggabungkan kedua mask
    mask = cv2.bitwise_or(mask_red, mask_orange)

    # Menemukan kontur pada mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur berdasarkan area untuk menghindari noise dan menghitung jumlah kapsul yang terdeteksi
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Menghitung jumlah kapsul
    capsule_count = 0
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        capsule_count += 1  # Tambahkan kapsul terdeteksi

    # Menampilkan jumlah kapsul terdeteksi
    print(f'Jumlah kapsul terdeteksi: {capsule_count}')

    # Mengatur pesan suara berdasarkan jumlah kapsul
    if capsule_count == 12:
        text_to_speak = "Sempurna."
    elif capsule_count < 12:
        text_to_speak = f'Kurang {12 - capsule_count}.'
    else:
        text_to_speak = f'Lebih {capsule_count - 12}.'

    # Mengonversi pesan ke suara
    tts = gTTS(text=text_to_speak, lang='id')
    audio_file = "capsule_count.mp3"
    tts.save(audio_file)

    # Memutar suara
    playsound(audio_file)
    os.remove(audio_file)  # Menghapus file audio setelah diputar

    # Menampilkan hasil pada gambar asli
    cv2.imshow('Gambar Asli dengan Deteksi Kapsul', image)

# Membuka webcam
cap = cv2.VideoCapture(0)

print("Tekan 'c' untuk mengambil gambar dan mendeteksi kapsul.")
print("Tekan 'q' untuk keluar.")

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break

    # Menampilkan frame dari webcam
    cv2.imshow('Webcam', frame)

    # Menunggu input dari keyboard
    key = cv2.waitKey(1) & 0xFF

    # Jika tombol 'c' ditekan, ambil gambar dan lakukan deteksi kapsul
    if key == ord('c'):
        print("Gambar diambil.")
        detect_capsules(frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    elif key == ord('q'):
        print("Keluar.")
        break

# Melepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
