import cv2
import numpy as np
import os
from gtts import gTTS
from playsound import playsound

# Path ke folder yang berisi gambar PNG blister
template_folder_path = "dataset/kemasan"  # Ganti dengan path yang sesuai
templates = []

# Muat setiap gambar PNG dalam folder sebagai template
for filename in os.listdir(template_folder_path):
    if filename.endswith(".png"):
        template_path = os.path.join(template_folder_path, filename)
        template = cv2.imread(template_path, 0)  # Memuat template dalam grayscale
        templates.append(template)

# Membuka akses ke webcam
cap = cv2.VideoCapture(0)

print("Tekan 'c' untuk mengambil gambar dan mendeteksi blitzer. Tekan 'q' untuk keluar.")

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca dari webcam.")
        break

    # Resize frame untuk mempercepat proses
    frame = cv2.resize(frame, (640, 480))  # Resize ke ukuran yang lebih kecil
    sample_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale

    # Tampilkan frame dari webcam
    cv2.imshow("Webcam", frame)

    # Tunggu input tombol
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Menyimpan gambar yang diambil
        sample_image = frame.copy()

        # Threshold untuk kecocokan template
        threshold = 0.7  # Menurunkan threshold untuk meningkatkan kemungkinan deteksi
        blitzer_count = 0

        # Lakukan template matching dengan setiap template yang ada di folder
        for template in templates:
            h, w = template.shape
            
            # Pastikan gambar yang diambil lebih besar dari template
            if sample_gray.shape[0] < h or sample_gray.shape[1] < w:
                print("Gambar terlalu kecil untuk template ini.")
                continue
            
            # Lakukan pencocokan template
            result = cv2.matchTemplate(sample_gray, template, cv2.TM_CCOEFF_NORMED)

            # Debug: Tampilkan hasil pencocokan
            cv2.imshow('Hasil Pencocokan', result)

            locations = np.where(result >= threshold)

            # Menggambar kotak di sekitar lokasi yang sesuai
            for pt in zip(*locations[::-1]):  # Balik koordinat x dan y
                cv2.rectangle(sample_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                blitzer_count += 1  # Increment count for each detected blitzer

        # Menampilkan gambar asli dengan kotak di sekitar blitzer yang terdeteksi
        cv2.imshow('Hasil Deteksi Blitzer (Gambar Asli)', sample_image)

        # Menampilkan jumlah blitzer terdeteksi
        print(f'Jumlah blitzer terdeteksi: {blitzer_count}')

        # Mengubah jumlah blitzer ke dalam suara
        text_to_speak = f'Ada {blitzer_count} blitzer'
        tts = gTTS(text=text_to_speak, lang='id')  # Menggunakan bahasa Indonesia
        audio_file = "blitzer_count.mp3"
        tts.save(audio_file)

        # Memutar suara
        playsound(audio_file)

        # Menghapus file audio setelah diputar
        os.remove(audio_file)

    elif key == ord('q'):
        # Keluar dari loop jika tombol 'q' ditekan
        break

# Membersihkan resource
cap.release()
cv2.destroyAllWindows()
