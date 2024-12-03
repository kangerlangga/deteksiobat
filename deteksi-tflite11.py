import os
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import threading
import mysql.connector
from datetime import datetime
import random
from db_config import DATABASE_CONFIG
from tensorflow.lite.python.interpreter import Interpreter

def speak_message(message, lang='id'):
    """Generate speech output using Google Text-to-Speech (gTTS)."""
    try:
        tts = gTTS(text=message, lang=lang)
        audio_file = "output.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        print(f"Error generating speech: {e}")

def save_to_database(blitzer, capsules, deficiency, status, created_by, modified_by):
    """Save detection results to the database."""
    connection = None
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = connection.cursor()
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        random_suffix = str(random.randint(100, 999))
        id_detections = f"Deteksi{timestamp}{random_suffix}"
        query = """
        INSERT INTO detections (id_detections, blitzer, kapsul, kekurangan, keterangan, created_by, modified_by, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (id_detections, blitzer, capsules, deficiency, status, created_by, modified_by, now, now)
        cursor.execute(query, data)
        connection.commit()
        print("Data berhasil disimpan ke database dengan ID:", id_detections)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def tflite_detect_realtime(modelpath, lblpath, MIN_CONF):
    """Real-time object detection with TFLite model."""
    if not os.path.exists(lblpath):
        raise FileNotFoundError(f"The label map file was not found at {lblpath}")
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam.")

    # Set high resolution and other webcam settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    cap.set(cv2.CAP_PROP_SATURATION, 50)

    print("Press 'q' to quit.")
    print("Press 'c' to capture the current frame and analyze detected objects.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Copy frame for real-time display
        display_frame = frame.copy()

        # Convert and preprocess image for TFLite model
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        count_kemasan = 0
        count_kapsul = 0

        for i in range(len(scores)):
            if (scores[i] > MIN_CONF) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i] * 100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(display_frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(display_frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                if object_name.lower() == "kemasan":
                    count_kemasan += 1
                elif object_name.lower() == "kapsul":
                    count_kapsul += 1

        expected_kapsul = count_kemasan * 12
        missing_kapsul = max(0, expected_kapsul - count_kapsul)
        status = "Sempurna" if missing_kapsul == 0 else "Cacat"

        info_text = f"Kemasan: {count_kemasan}, Kapsul: {count_kapsul}, Kekurangan: {missing_kapsul}, Status: {status}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Sistem Pendeteksi Kekurangan Obat Kapsida HS', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save frame with detections for display
            captured_frame = display_frame.copy()

            # Save data to the database
            save_to_database(count_kemasan, count_kapsul, missing_kapsul, status, "User", "User")

            # Show captured frame in a separate window
            cv2.imshow('Hasil Deteksi', captured_frame)

            # Speak the results
            if missing_kapsul > 0:
                threading.Thread(target=speak_message, args=(f"Kurang {missing_kapsul}",)).start()
            else:
                threading.Thread(target=speak_message, args=("Sempurna",)).start()

    cap.release()
    cv2.destroyAllWindows()

# Path configurations
PATH_TO_MODEL = 'dataset/detect.tflite'
PATH_TO_LABELS = 'labelmap.txt'
MIN_CONF = 0.01

# Run the detection
tflite_detect_realtime(PATH_TO_MODEL, PATH_TO_LABELS, MIN_CONF)
