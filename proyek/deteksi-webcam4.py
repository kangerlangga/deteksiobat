import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from gtts import gTTS
import playsound
import socket
import mysql.connector
from datetime import datetime
import random
from db_config import DATABASE_CONFIG

# Fungsi untuk mengecek koneksi internet
def check_internet_connection():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False

# Fungsi untuk menyimpan data ke database
def save_to_database(blitzer, capsules, deficiency, status, created_by, modified_by):
    print("Memulai penyimpanan data ke database...")
    connection = None
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        print("Koneksi ke database berhasil.")
        cursor = connection.cursor()

        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        random_suffix = str(random.randint(100, 999))
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

# Periksa koneksi internet
if not check_internet_connection():
    print("Tidak ada koneksi internet. Program akan berhenti.")
    exit()

# Load configuration and set up from Model Zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model.pth"
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Define class labels and their respective colors
class_labels = {0: "Kemasan", 1: "Kapsul"}
class_colors = {0: (255, 0, 0), 1: (0, 255, 0)}

# Open the camera feed
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Tidak dapat mengakses kamera")
    exit()

max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi maksimal yang didukung kamera: {max_width}x{max_height}")

desired_width = 1280
desired_height = 720

if max_width < desired_width or max_height < desired_height:
    desired_width = max_width
    desired_height = max_height
    print(f"Resolusi lebih kecil dari 1280x720, menggunakan resolusi maksimal {desired_width}x{desired_height}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi kamera setelah diatur: {frame_width}x{frame_height}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    cv2.imshow("Sistem Pendeteksi Jumlah Kapsul Kapsida HS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        outputs = predictor(frame)
        instances = outputs["instances"]

        kapsul_count = 0
        kemasan_count = 0

        if instances.has("pred_boxes") and instances.has("pred_classes"):
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                label = class_labels.get(cls, "Unknown")
                color = class_colors.get(cls, (0, 255, 255))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if label == "Kapsul":
                    kapsul_count += 1
                elif label == "Kemasan":
                    kemasan_count += 1

        required_kapsul = kemasan_count * 12
        shortfall = required_kapsul - kapsul_count

        if shortfall > 0:
            message = f"Kurang {shortfall}"
            status = "Cacat"
        else:
            message = "Sempurna"
            status = "Sempurna"

        tts = gTTS(message, lang="id")
        tts.save("output.mp3")
        playsound.playsound("output.mp3")
        os.remove("output.mp3")

        cv2.putText(frame, f"Kapsul: {kapsul_count};  Kemasan: {kemasan_count}; {message};", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hasil Deteksi", frame)

        # Save detection data to the database
        save_to_database(kemasan_count, kapsul_count, shortfall, status, "Kapsida Count", "Kapsida Count")

cap.release()
cv2.destroyAllWindows()
