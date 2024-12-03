import cv2

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('cascade_blitzer.xml')

# Buka kamera (webcam)
cap = cv2.VideoCapture(0)  # 0 untuk default webcam, ganti jika Anda menggunakan kamera eksternal

if not cap.isOpened():
    print("Tidak dapat mengakses kamera")
    exit()

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Jika gagal menangkap frame
    if not ret:
        print("Gagal membaca frame")
        break

    # Konversi ke grayscale untuk deteksi yang lebih akurat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Tampilkan frame
    cv2.imshow('Deteksi Blitzer', frame)

    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepas kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
