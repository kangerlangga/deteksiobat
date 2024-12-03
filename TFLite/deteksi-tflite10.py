import os
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import threading
from tensorflow.lite.python.interpreter import Interpreter
import mysql.connector
from datetime import datetime
import random
from db_config import DATABASE_CONFIG

# Fungsi untuk menghasilkan suara
def speak_message(status, missing_count):
    try:
        if missing_count > 0:
            message = f"Kurang {missing_count}."
        else:
            message = f"Sempurna."

        tts = gTTS(message, lang='id')
        temp_filename = f"status_{random.randint(1000, 9999)}.mp3"
        tts.save(temp_filename)
        playsound(temp_filename)
        os.remove(temp_filename)
    except Exception as e:
        print(f"Error menghasilkan suara: {e}")

# Fungsi untuk mendeteksi kemasan
def detect_kemasan(frame):
    """
    Mendeteksi kemasan menggunakan filter warna (mirip deteksi blitzer).
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 180])
    upper_gray = np.array([180, 30, 230])
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 25, 255])
    mask_gray = cv2.inRange(hsv_frame, lower_gray, upper_gray)
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)
    mask_kemasan = cv2.bitwise_or(mask_gray, mask_white)
    contours, _ = cv2.findContours(mask_kemasan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_kemasan = [c for c in contours if cv2.contourArea(c) > 500]  # Filter area kecil

    kemasan_count = 0
    for contour in valid_kemasan:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Warna biru
        kemasan_count += 1

    return kemasan_count, frame

# Fungsi untuk mendeteksi kapsul menggunakan model TFLite dengan filter warna
def detect_capsules_with_tflite(frame, interpreter, input_details, output_details, labels, min_conf=0.5):
    imH, imW, _ = frame.shape
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Persiapan data untuk TFLite
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Jalankan model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Ambil hasil deteksi
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    kapsul_count = 0
    detected_boxes = []  # Menyimpan bounding box dari model untuk validasi warna

    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            if object_name.lower() == "kapsul":
                kapsul_count += 1
                detected_boxes.append((xmin, ymin, xmax, ymax))
                # Gunakan warna yang sama (misalnya hijau) untuk bounding box dari TFLite
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Warna hijau

    # Tidak ada lagi filter warna yang diterapkan di sini
    return kapsul_count, frame

# Fungsi untuk menyimpan data ke database
def save_to_database(kemasan_count, kapsul_count, deficiency, status, created_by, modified_by):
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = connection.cursor()

        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        random_suffix = str(random.randint(100, 999))
        id_detections = f"Deteksi{timestamp}{random_suffix}"

        query = """
        INSERT INTO detections (id_detections, blitzer, kapsul, kekurangan, keterangan, created_by, modified_by, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (id_detections, kemasan_count, kapsul_count, deficiency, status, created_by, modified_by, now, now)
        cursor.execute(query, data)
        connection.commit()
        print(f"Data berhasil disimpan ke database dengan ID: {id_detections}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Fungsi utama untuk deteksi real-time
def detect_realtime(modelpath, lblpath, min_conf=0.5):
    # Load model TFLite
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load label map
    if not os.path.exists(lblpath):
        raise FileNotFoundError(f"Label file tidak ditemukan: {lblpath}")
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Kamera gagal dibuka.")

    print("Tekan 'q' untuk keluar.")
    print("Tekan 'c' untuk menyimpan data deteksi ke database.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        realtime_frame = frame.copy()

        # Deteksi kemasan
        kemasan_count, kemasan_frame = detect_kemasan(realtime_frame)

        # Deteksi kapsul menggunakan TFLite dan filter warna
        kapsul_count, kapsul_frame = detect_capsules_with_tflite(
            frame=realtime_frame,
            interpreter=interpreter,
            input_details=input_details,
            output_details=output_details,
            labels=labels,
            min_conf=min_conf
        )

        # Hitung kekurangan dan status
        expected_kapsul = kemasan_count * 12
        missing_kapsul = max(0, expected_kapsul - kapsul_count)
        status = "Sempurna" if missing_kapsul == 0 else "Cacat"

        # Tambahkan informasi ke frame utama
        info_text = f"Kemasan: {kemasan_count}, Kapsul: {kapsul_count}, Kekurangan: {missing_kapsul}, Status: {status}"
        cv2.putText(realtime_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Sistem Pendeteksi Kekurangan Obat Kapsida HS', realtime_frame)

        # Tindakan berdasarkan input keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"Data disimpan: Kemasan={kemasan_count}, Kapsul={kapsul_count}, Kekurangan={missing_kapsul}, Status={status}")
            save_to_database(kemasan_count, kapsul_count, missing_kapsul, status, "User", "User")

            # Menyimpan frame hasil deteksi
            detection_frame = kapsul_frame.copy()

            # Tampilkan frame hasil capture di jendela terpisah
            cv2.imshow('Hasil Deteksi (Capture)', detection_frame)

            # Menjalankan suara status di thread terpisah
            threading.Thread(target=speak_message, args=(status, missing_kapsul)).start()

    cap.release()
    cv2.destroyAllWindows()

# Jalankan deteksi
PATH_TO_MODEL = 'dataset/detect.tflite'
PATH_TO_LABELS = 'labelmap.txt'
MIN_CONFIDENCE = 0.5

detect_realtime(PATH_TO_MODEL, PATH_TO_LABELS, MIN_CONFIDENCE)