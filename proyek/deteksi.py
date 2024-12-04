import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from google.colab.patches import cv2_imshow

# 1. Membuat konfigurasi default
cfg = get_cfg()

# 2. Menggunakan model yang sudah dilatih dan file .pth
cfg.MODEL.WEIGHTS = "model.pth"  # Path ke model .pth yang Anda miliki
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold deteksi objek

# Menetapkan model yang digunakan (misalnya Faster R-CNN atau Mask R-CNN)
cfg.MODEL.DEVICE = "cuda"  # Atau "cpu" jika tidak menggunakan GPU
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Misalnya 80 kelas (untuk COCO), Anda bisa menyesuaikan ini jika perlu

# Dataset dan Metadata (meskipun Anda tidak menggunakan dataset yang terdaftar, Anda bisa mengaturnya secara manual)
cfg.DATASETS.TEST = ()  # Kosongkan dataset karena kita tidak menggunakannya
test_metadata = MetadataCatalog.get("kapsida")  # Anda dapat menyesuaikan metadata atau membuat metadata dummy

# 3. Inisialisasi predictor untuk inferensi
predictor = DefaultPredictor(cfg)

# 4. Membaca gambar dan melakukan deteksi
image_path = 'test.jpg'  # Ganti dengan path gambar Anda
im = cv2.imread(image_path)

# Lakukan inferensi pada gambar
outputs = predictor(im)

# 5. Visualisasi hasil deteksi
v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# 6. Menampilkan hasil deteksi
cv2_imshow(out.get_image()[:, :, ::-1])  # Menampilkan gambar dengan deteksi objek
