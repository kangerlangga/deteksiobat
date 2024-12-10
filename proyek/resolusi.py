import cv2

def check_camera_resolution():
    print("Mendeteksi resolusi kamera pada ID yang tersedia...")
    for camera_id in range(10):  # Coba ID kamera dari 0 sampai 9
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"Kamera ditemukan pada ID: {camera_id}")

            # Paksa resolusi ke 1280x720
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Periksa resolusi yang berhasil diatur
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Resolusi setelah diatur (ID {camera_id}): {width}x{height}")
        else:
            print(f"Tidak ada kamera pada ID: {camera_id}")
        cap.release()

if __name__ == "__main__":
    check_camera_resolution()
