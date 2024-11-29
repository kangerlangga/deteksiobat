import cv2
import numpy as np
import tensorflow as tf

def load_labels(label_path):
    """Muat label dari file."""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def run_inference(interpreter, input_data):
    """Lakukan inferensi pada model TFLite."""
    input_index = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Ekstrak hasil inferensi
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Kotak deteksi
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Kelas objek
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    return boxes, class_ids, scores

def apply_nms(boxes, scores, max_detections=150, iou_threshold=0.5):
    """Terapkan Non-Maximum Suppression untuk hasil deteksi."""
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=max_detections, iou_threshold=iou_threshold
    )
    filtered_boxes = np.array([boxes[i] for i in indices])
    filtered_scores = np.array([scores[i] for i in indices])
    return filtered_boxes, filtered_scores

def preprocess_image(image, height, width):
    """Preproses gambar untuk input ke model."""
    resized_image = cv2.resize(image, (width, height))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb_image, axis=0).astype(np.float32)
    input_data = (input_data - 127.5) / 127.5  # Normalisasi
    return input_data

def detect_realtime_tflite(model_path, label_path, min_conf=0.5):
    """Lakukan deteksi real-time menggunakan TFLite."""
    # Muat model dan label
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    labels = load_labels(label_path)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]

    # Buka webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka webcam.")
        return

    print("Tekan 'q' untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap frame.")
            break
        
        imH, imW, _ = frame.shape
        
        # Preproses gambar
        input_data = preprocess_image(frame, input_height, input_width)
        
        # Jalankan inferensi
        boxes, class_ids, scores = run_inference(interpreter, input_data)

        # Terapkan NMS untuk menyaring hasil
        filtered_boxes, filtered_scores = apply_nms(boxes, scores, max_detections=150)

        # Tampilkan hasil deteksi
        for i, box in enumerate(filtered_boxes):
            if filtered_scores[i] >= min_conf:
                ymin, xmin, ymax, xmax = box
                ymin = int(max(1, ymin * imH))
                xmin = int(max(1, xmin * imW))
                ymax = int(min(imH, ymax * imH))
                xmax = int(min(imW, xmax * imW))
                
                class_name = labels[int(class_ids[i])]
                label = f"{class_name}: {int(filtered_scores[i] * 100)}%"
                
                # Gambar kotak dan label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Tampilkan frame dengan deteksi
        cv2.imshow("Real-Time Detection", frame)

        # Keluar jika 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan deteksi real-time
MODEL_PATH = "dataset/detect.tflite"
LABEL_PATH = "labelmap.txt"
detect_realtime_tflite(MODEL_PATH, LABEL_PATH)
