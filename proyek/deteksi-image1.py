import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from datetime import datetime

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

# Path to the input image
image_path = "test.jpeg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"File {image_path} tidak ditemukan!")

# Read the input image
image = cv2.imread(image_path)

# Perform object detection
outputs = predictor(image)
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
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Add label text above the bounding box
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)  # Increase font size (1) and thickness (3)

        # Count objects by class
        if label == "Kapsul":
            kapsul_count += 1
        elif label == "Kemasan":
            kemasan_count += 1

# Add text to show the count of detected objects on the image
cv2.putText(image, f"Kapsul: {kapsul_count}  Kemasan: {kemasan_count}", (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 15, cv2.LINE_AA)  # Horizontal position for count

# Resize the image to fit the screen
screen_width = 1280  # Adjust according to your screen width
screen_height = 720  # Adjust according to your screen height
height, width, _ = image.shape

if width > screen_width or height > screen_height:
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)  # Use the smallest scale to fit
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Create a folder 'hasil' if it doesn't exist
output_folder = "hasil"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Generate a unique filename using current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_folder, f"result_{timestamp}.jpg")

# Save the resulting image
cv2.imwrite(output_path, image)
print(f"Hasil deteksi disimpan di {output_path}")

# Display the resulting image
cv2.imshow("Sistem Pendeteksi Jumlah Obat Kapsida HS", image)

# Display counts in terminal
print(f"Jumlah Kapsul: {kapsul_count}")
print(f"Jumlah Kemasan: {kemasan_count}")

cv2.waitKey(0)
cv2.destroyAllWindows()
