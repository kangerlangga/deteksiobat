import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from gtts import gTTS
import playsound

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
class_colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Warna untuk setiap kelas: biru untuk kemasan, hijau untuk kapsul

# Open the camera feed
cap = cv2.VideoCapture(1)  # Gunakan ID kamera yang sesuai jika menggunakan DroidCam atau perangkat lain

if not cap.isOpened():
    print("Tidak dapat mengakses kamera")
    exit()

# Set camera resolution to 720p (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get and display the resolution to verify
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi kamera setelah diatur: {frame_width}x{frame_height}")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # Display the webcam feed
    cv2.imshow("Webcam Real-Time", frame)

    # Check for keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit if 'q' is pressed
        break
    elif key == ord('c'):  # Capture frame and process detection if 'c' is pressed
        # Perform object detection on the captured frame
        outputs = predictor(frame)
        instances = outputs["instances"]

        # Variables to count detected objects
        kapsul_count = 0
        kemasan_count = 0

        # Draw bounding boxes, labels, and other information on the captured frame
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # Font scale 0.7, thickness 2

                # Count objects by class
                if label == "Kapsul":
                    kapsul_count += 1
                elif label == "Kemasan":
                    kemasan_count += 1

        # Calculate the shortage of capsules
        required_kapsul = kemasan_count * 12
        shortfall = required_kapsul - kapsul_count

        # Determine the message based on the shortfall
        if shortfall > 0:
            message = f"Kurang {shortfall}"
        else:
            message = "Sempurna"

        # Generate and play the audio
        tts = gTTS(message, lang="id")
        tts.save("output.mp3")
        playsound.playsound("output.mp3")
        os.remove("output.mp3")  # Remove the audio file after playing

        # Add text to show the count of detected objects and the shortfall message
        cv2.putText(frame, f"Kapsul: {kapsul_count};  Kemasan: {kemasan_count}; {message};", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, message, (10 + 200, 80),  # Letakkan message di samping Kemasan
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the detection result in a new window
        cv2.imshow("Deteksi Objek dari Capture", frame)

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
