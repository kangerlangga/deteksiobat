import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

# Load the model
model = torch.load("model.pth")
model.eval()  # Set the model to evaluation mode

# Preprocessing (sesuaikan dengan preprocessing model Detectron2 Anda)
transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Fungsi untuk melakukan inferensi pada gambar
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Tambahkan dimensi batch
    with torch.no_grad():
        predictions = model(image)  # Melakukan prediksi
    return predictions

# Contoh pemakaian
predictions = infer("test.jpeg")
print(predictions)
