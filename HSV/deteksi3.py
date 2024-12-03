import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
import threading
import mysql.connector
from datetime import datetime
import random
from db_config import DATABASE_CONFIG  # Import konfigurasi database

# Fungsi untuk mendeteksi kapsul pada gambar yang diambil
def detect_capsules(image):
    sample_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    lower_orange = np.array([11, 70, 50])
    upper_orange = np.array([25, 255, 255])
    mask_red = cv2.inRange(sample_hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(sample_hsv, lower_orange, upper_orange)
    mask_capsule = cv2.bitwise_or(mask_red, mask_orange)
    contours, _ = cv2.findContours(mask_capsule, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_capsules = [contour for contour in contours if cv2.contourArea(contour) > 100]
    capsule_count = 0
    for contour in valid_capsules:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        capsule_count += 1
    return capsule_count, image

# Fungsi untuk mendeteksi blitzer pada gambar yang diambil
def detect_blitzers(image):
    sample_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 180])
    upper_gray = np.array([180, 30, 230])
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 25, 255])
    mask_gray = cv2.inRange(sample_hsv, lower_gray, upper_gray)
    mask_white = cv2.inRange(sample_hsv, lower_white, upper_white)
    mask_blitzer = cv2.bitwise_or(mask_gray, mask_white)
    contours, _ = cv2.findContours(mask_blitzer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blitzers = [contour for contour in contours if cv2.contourArea(contour) > 500]
    blitzer_count = 0
    for contour in valid_blitzers:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        blitzer_count += 1
    return blitzer_count, image

# Fungsi untuk menghitung kekurangan dan keterangan
def calculate_deficiency_and_status(capsule_count, blitzer_count):
    total_capsules = (blitzer_count * 12) + capsule_count
    deficiency = max(0, (blitzer_count * 12) - capsule_count)
    status = "Sempurna" if deficiency == 0 else "Cacat"
    return deficiency, status

# Fungsi untuk menyimpan data deteksi ke database
def save_to_database(blitzer, capsules, deficiency, status, created_by, modified_by):
    print("Memulai penyimpanan data ke database...")
    connection = None
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        print("Koneksi ke database berhasil.")
        cursor = connection.cursor()

        # Generate id_detections dengan format DeteksiYYYYMMDDHHMMSSXXX
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')  # Format tanggal dan waktu
        random_suffix = str(random.randint(100, 999))  # Tiga digit angka acak
        id_detections = f"Deteksi{timestamp}{random_suffix}"

        query = """
        INSERT INTO detections (id_detections, blitzer, kapsul, kekurangan, keterangan, created_by, modified_by, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (id_detections, blitzer, capsules, deficiency, status, created_by, modified_by, now, now)
        cursor.execute(query, data)
        connection.commit()
        print("Data berhasil disimpan ke database dengan ID:", id_detections)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("Koneksi ke database ditutup.")

# Fungsi untuk memutar suara
def play_sound(text):
    tts = gTTS(text=text, lang='id')
    audio_file = "detection_result.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

# Membuka akses ke webcam
cap = cv2.VideoCapture(0)
print("Tekan 'c' untuk mengambil gambar dan mendeteksi kapsul serta blitzer.")
print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break
    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print("Gambar diambil.")
        capsule_count, annotated_image = detect_capsules(frame.copy())
        print(f'Jumlah kapsul terdeteksi: {capsule_count}')
        blitzer_count, annotated_image = detect_blitzers(annotated_image)
        print(f'Jumlah kemasan terdeteksi: {blitzer_count}')
        deficiency, status = calculate_deficiency_and_status(capsule_count, blitzer_count)
        print(f'Kekurangan: {deficiency}, Status: {status}')
        save_to_database(blitzer_count, capsule_count, deficiency, status, "Sistem Deteksi Obat", "Sistem Deteksi Obat")
        cv2.imshow('Hasil Deteksi Kapsul dan Blitzer', annotated_image)
        text_to_speak = f'Terdeteksi {capsule_count} kapsul, {blitzer_count} kemasan. Kekurangan {deficiency} kapsul. Status {status}.'
        threading.Thread(target=play_sound, args=(text_to_speak,)).start()
    elif key == ord('q'):
        print("Keluar.")
        break

cap.release()
cv2.destroyAllWindows()
