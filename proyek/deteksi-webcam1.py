import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Load configuration and set up from Model Zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Gunakan konfigurasi YAML dari Model Zoo
cfg.MODEL.WEIGHTS = "model.pth"  # Path ke model .pth Anda
cfg.MODEL.DEVICE = "cpu"  # Gunakan "cuda" jika ingin menggunakan GPU
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold deteksi
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Pastikan model dilatih dengan 2 kelas: kapsul dan kemasan

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Define class labels and their respective colors
class_labels = {0: "Kemasan", 1: "Kapsul"}  # Pastikan label sesuai urutan kelas saat training
class_colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Warna untuk setiap kelas: hijau untuk kapsul, biru untuk kemasan

# Open the camera feed (Use 0 for default webcam or replace with DroidCam ID like 1, 2, etc.)
cap = cv2.VideoCapture(0)  # Jika menggunakan DroidCam, pastikan DroidCam sudah terhubung dan gunakan ID yang benar

if not cap.isOpened():
    print("Tidak dapat mengakses kamera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal membaca frame")
        break

    # Perform object detection
    outputs = predictor(frame)
    instances = outputs["instances"]

    # Variables to count detected objects
    kapsul_count = 0
    kemasan_count = 0

    # Draw bounding boxes, labels, and other information on the image
    if instances.has("pred_boxes") and instances.has("pred_classes"):
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            label = class_labels.get(cls, "Unknown")
            color = class_colors.get(cls, (0, 255, 255))  # Default to yellow if class not found

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Add label text above the bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Count objects by class
            if label == "Kapsul":
                kapsul_count += 1
            elif label == "Kemasan":
                kemasan_count += 1

    # Menyesuaikan ukuran font dan ketebalan teks agar tidak terlalu besar
    font_scale = 1  # Ukuran font lebih kecil
    thickness = 2    # Ketebalan garis teks

    # Add text to show the count of detected objects on the image
    cv2.putText(frame, f"Kapsul: {kapsul_count}  Kemasan: {kemasan_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Deteksi Objek Real-Time", frame)

    # Press 'q' to quit the live video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
