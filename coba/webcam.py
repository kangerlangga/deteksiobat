import cv2

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Mengecek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Kamera tidak dapat dibuka.")
    exit()

# Membuat variabel counter untuk penamaan file gambar yang di-capture
counter = 0

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    
    # Jika frame tidak terbaca, keluar dari loop
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break
    
    # Menampilkan frame
    cv2.imshow("Webcam", frame)
    
    # Menunggu input dari keyboard
    key = cv2.waitKey(1) & 0xFF
    
    # Jika tombol 'c' ditekan, capture gambar
    if key == ord('c'):
        img_name = f"capture_{counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"Gambar disimpan: {img_name}")
        counter += 1
    
    # Jika tombol 'q' ditekan, keluar dari loop
    elif key == ord('q'):
        print("Program selesai.")
        break

# Melepaskan kamera dan menutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
