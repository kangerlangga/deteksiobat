import cv2
import numpy as np
import onnxruntime as ort
import requests
from io import BytesIO
from PIL import Image

# Fungsi untuk mengunduh gambar dari URL
def download_image(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Fungsi untuk memproses gambar sebelum memberikan input ke model ONNX
def preprocess_image(image, target_size=(640, 640)):
    # Mengubah gambar ke format yang dapat diterima oleh model
    image = image.convert("RGB")  # Pastikan gambar dalam format RGB
    image = image.resize(target_size)  # Ubah ukuran gambar sesuai dengan yang diinginkan

    # Mengonversi gambar ke numpy array
    image = np.array(image)

    # Normalisasi: ubah pixel menjadi nilai [0,1]
    image = image.astype(np.float32) / 255.0

    # Konversi ke format CHW (channel, height, width)
    image = np.transpose(image, (2, 0, 1))  # Ubah dari HWC ke CHW

    # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=0)  # Jadi ukuran: (1, 3, 640, 640)

    return image

# Fungsi untuk mendeteksi objek
def detect_objects(model_path, image_path):
    # Unduh gambar (dari URL atau file lokal)
    image = None
    if image_path.startswith('http'):
        image = download_image(image_path)
    else:
        image = cv2.imread(image_path)

    if image is None:
        print("Gambar tidak ditemukan atau gagal diunduh!")
        return

    # Preprocessing gambar
    image_input = preprocess_image(image)

    # Menghapus dimensi batch yang berlebih
    image_input = np.squeeze(image_input, axis=0)  # Hapus dimensi batch (jadi ukuran: (3, 640, 640))

    # Memuat model ONNX
    session = ort.InferenceSession(model_path)

    # Nama input dan output
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Melakukan inferensi
    outputs = session.run(output_names, {input_name: image_input})

    # Output pertama adalah box, kedua adalah skor, ketiga adalah label
    boxes = outputs[0]
    scores = outputs[1]
    labels = outputs[2]

    # Visualisasi hasil deteksi
    visualize_detections(image, boxes, scores, labels)

# Fungsi untuk menvisualisasikan hasil deteksi
def visualize_detections(image, boxes, scores, labels, threshold=0.5):
    # Lakukan pemrosesan untuk menampilkan kotak pembatas hanya untuk deteksi dengan skor lebih tinggi dari threshold
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            label = labels[i]
            score = scores[i]

            # Gambar kotak pembatas pada gambar
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # Gambar label dan skor
            cv2.putText(image, f"Label: {label}, {score:.2f}", 
                        (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)

    # Tampilkan gambar dengan deteksi menggunakan OpenCV
    cv2.imshow("Deteksi Objek", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main program untuk mendeteksi objek dari gambar
if __name__ == "__main__":
    model_path = "coba1.onnx"  # Ganti dengan path model ONNX Anda
    image_path = "https://edorusyanto.wordpress.com/wp-content/uploads/2015/05/lalin-jakarta-padat_1.jpg"  # Ganti dengan path ke gambar lokal atau URL gambar

    detect_objects(model_path, image_path)
