import cv2

# Coba berbagai resolusi
desired_width = 1280
desired_height = 720

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Tidak dapat mengakses kamera")
else:
    # Mencetak resolusi kamera yang dapat diterima
    print("Resolusi yang didukung kamera:")
    for width, height in [(1920, 1080), (1280, 720), (640, 480)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolusi yang diset: {frame_width}x{frame_height}")
    
cap.release()
