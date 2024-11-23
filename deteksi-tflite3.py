import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

def tflite_detect_realtime(modelpath, lblpath, min_conf=0.5):
    """
    Perform object detection in real-time using a webcam and a TFLite model.
    """
    # Load the label map into memory
    if not os.path.exists(lblpath):
        raise FileNotFoundError(f"The label map file was not found at {lblpath}")
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam.")

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Prepare the image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class indices
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

        count_kemasan = 0
        count_kapsul = 0

        # Loop over all detections and process them if confidence is above the threshold
        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i]*100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-label_size[1]-10), (xmin+label_size[0], label_ymin+base_line-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Count objects by type
                if object_name.lower() == "kemasan":
                    count_kemasan += 1
                elif object_name.lower() == "kapsul":
                    count_kapsul += 1

        # Display the resulting frame
        cv2.putText(frame, f"Kemasan: {count_kemasan}, Kapsul: {count_kapsul}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Object Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Path configurations
PATH_TO_MODEL = 'dataset/custom_model_lite/detect.tflite'
PATH_TO_LABELS = 'labelmap.txt'

# Confidence threshold
min_conf_threshold = 0.5

# Run the model for real-time detection
tflite_detect_realtime(
    modelpath=PATH_TO_MODEL,
    lblpath=PATH_TO_LABELS,
    min_conf=min_conf_threshold
)
