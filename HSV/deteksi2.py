import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
import threading

# Fungsi untuk mendeteksi kapsul pada gambar yang diambil
def detect_capsules(image):
    # Konversi gambar ke HSV untuk deteksi warna kapsul
    sample_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rentang warna untuk mendeteksi warna merah dan oranye (kapsul)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    lower_orange = np.array([11, 70, 50])
    upper_orange = np.array([25, 255, 255])

    # Mask untuk warna merah dan oranye
    mask_red = cv2.inRange(sample_hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(sample_hsv, lower_orange, upper_orange)

    # Gabungkan kedua mask untuk kapsul
    mask_capsule = cv2.bitwise_or(mask_red, mask_orange)

    # Temukan kontur kapsul
    contours, _ = cv2.findContours(mask_capsule, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur kapsul berdasarkan area untuk menghindari noise
    valid_capsules = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Hitung dan tandai kapsul
    capsule_count = 0
    for contour in valid_capsules:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Bounding box hijau untuk kapsul
        capsule_count += 1  # Tambahkan kapsul yang terdeteksi

    return capsule_count, image

# Fungsi untuk mendeteksi blitzer pada gambar yang diambil
def detect_blitzers(image):
    # Konversi gambar ke HSV untuk deteksi warna blitzer
    sample_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rentang warna untuk abu-abu dan putih (blitzer)
    lower_gray = np.array([0, 0, 180])
    upper_gray = np.array([180, 30, 230])
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 25, 255])

    # Mask untuk abu-abu dan putih
    mask_gray = cv2.inRange(sample_hsv, lower_gray, upper_gray)
    mask_white = cv2.inRange(sample_hsv, lower_white, upper_white)

    # Gabungkan mask abu-abu dan putih untuk blitzer
    mask_blitzer = cv2.bitwise_or(mask_gray, mask_white)

    # Temukan kontur blitzer
    contours, _ = cv2.findContours(mask_blitzer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur blitzer berdasarkan area untuk menghindari noise
    valid_blitzers = [contour for contour in contours if cv2.contourArea(contour) > 500]

    # Hitung dan tandai blitzer
    blitzer_count = 0
    for contour in valid_blitzers:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Bounding box biru untuk blitzer
        blitzer_count += 1  # Tambahkan blitzer yang terdeteksi

    return blitzer_count, image

# Fungsi untuk memutar suara
def play_sound(text):
    tts = gTTS(text=text, lang='id')
    audio_file = "detection_result.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)  # Menghapus file audio setelah diputar

# Membuka akses ke webcam
cap = cv2.VideoCapture(0)

print("Tekan 'c' untuk mengambil gambar dan mendeteksi kapsul serta blitzer.")
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

    # Jika tombol 'c' ditekan, ambil gambar dan lakukan deteksi kapsul dan blitzer
    if key == ord('c'):
        print("Gambar diambil.")
        
        # Deteksi kapsul
        capsule_count, annotated_image = detect_capsules(frame.copy())
        print(f'Jumlah kapsul terdeteksi: {capsule_count}')

        # Deteksi blitzer
        blitzer_count, annotated_image = detect_blitzers(annotated_image)
        print(f'Jumlah blitzer terdeteksi: {blitzer_count}')

        # Tampilkan hasil deteksi pada gambar asli
        cv2.imshow('Hasil Deteksi Kapsul dan Blitzer', annotated_image)

        # Tentukan pesan suara berdasarkan hasil deteksi
        text_to_speak = f'Terdeteksi {capsule_count} kapsul dan {blitzer_count} blitzer.'

        # Memutar suara dalam thread terpisah
        threading.Thread(target=play_sound, args=(text_to_speak,)).start()

    # Jika tombol 'q' ditekan, keluar dari loop
    elif key == ord('q'):
        print("Keluar.")
        break

# Melepaskan sumber daya
cap.release()
cv2.destroyAllWindows()
