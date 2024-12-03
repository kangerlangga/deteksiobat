import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Definisikan model CNN (CapsulePackagingModel)
class CapsulePackagingModel(torch.nn.Module):
    def __init__(self):
        super(CapsulePackagingModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        
        # Hitung ukuran output setelah konvolusi
        self.flatten_size = self._get_flatten_size((224, 224))  # Sesuaikan dengan ukuran input

        self.fc1 = torch.nn.Linear(self.flatten_size, 512)  # Sesuaikan ukuran input
        self.fc2 = torch.nn.Linear(512, 1)  # Misalnya output satu nilai untuk deteksi objek

    def _get_flatten_size(self, image_size):
        # Cek ukuran output setelah konvolusi
        x = torch.zeros(1, 3, *image_size)  # Gambar dummy untuk menghitung ukuran
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return int(np.prod(x.size()))  # Jumlah elemen setelah konvolusi (flattened)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten sebelum masuk ke fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output deteksi
        return x

# Kelas untuk deteksi objek
class CapsulePackagingDetector:
    def __init__(self, model_path):
        self.model = CapsulePackagingModel()
        # Muat model menggunakan strict=False untuk mengabaikan kunci yang tidak cocok
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)  # Mengabaikan kunci yang tidak cocok
        self.model.eval()

    def detect(self, image):
        # Transformasi input image
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),  # Sesuaikan ukuran input jika perlu
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisasi
        ])
        image_tensor = transform(image).unsqueeze(0)  # Tambahkan batch dimension
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs

# Inisialisasi detektor
model_path = "model.pth"  # Ganti dengan path ke file model Anda
detector = CapsulePackagingDetector(model_path)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # Sesuaikan dengan indeks kamera Anda
if not cap.isOpened():
    print("Error: Webcam tidak terdeteksi!")
    exit()

print("Menjalankan deteksi real-time. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam!")
        break
    
    # Konversi frame OpenCV (BGR) ke PIL (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Deteksi objek
    detections = detector.detect(pil_image)
    
    # Interpretasi hasil deteksi
    # Misalnya jika model menghasilkan satu nilai untuk deteksi objek:
    detection_result = detections.item()  # Ambil nilai deteksi dari output
    
    # Tambahkan teks pada frame
    cv2.putText(frame, f"Deteksi: {detection_result:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilkan frame di jendela
    cv2.imshow("Deteksi Objek", frame)
    
    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya
cap.release()
cv2.destroyAllWindows()