# Import required libraries
import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt

def tflite_detect_single_image(modelpath, imgpath, lblpath, min_conf=0.5, savepath='./results', txt_only=False):
    """
    Perform object detection on a single image using a TFLite model.
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

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(imgpath)
    if image is None:
        raise FileNotFoundError(f"The image file was not found at {imgpath}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class indices
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

    detections = []
    count_kemasan = 0
    count_kapsul = 0

    # Loop over all detections and process them if confidence is above the threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i]*100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin-label_size[1]-10), (xmin+label_size[0], label_ymin+base_line-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            # Count objects by type
            if object_name.lower() == "kemasan":
                count_kemasan += 1
            elif object_name.lower() == "kapsul":
                count_kapsul += 1

    # Display the image with detection boxes
    if not txt_only:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 16))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    # Save detection results to a text file (for mAP calculation)
    os.makedirs(savepath, exist_ok=True)
    base_fn, _ = os.path.splitext(os.path.basename(imgpath))
    txt_result_fn = f'{base_fn}.txt'
    txt_savepath = os.path.join(savepath, txt_result_fn)

    if txt_only:
        with open(txt_savepath, 'w') as f:
            for detection in detections:
                f.write(f'{detection[0]} {detection[1]:.4f} {detection[2]} {detection[3]} {detection[4]} {detection[5]}\n')

    # Return counts of detected objects
    return count_kemasan, count_kapsul


# Path configurations
PATH_TO_IMAGES = 'coba/b1.jpeg'  # Path ke gambar
PATH_TO_MODEL = 'dataset/custom_model_lite/detect.tflite'
PATH_TO_LABELS = 'labelmap.txt'

# Confidence threshold
min_conf_threshold = 0.5

# Run the model and count objects
count_kemasan, count_kapsul = tflite_detect_single_image(
    modelpath=PATH_TO_MODEL,
    imgpath=PATH_TO_IMAGES,
    lblpath=PATH_TO_LABELS,
    min_conf=min_conf_threshold,
    savepath='./results',
    txt_only=False
)

# Display the results
print("Hasil Deteksi:")
print(f"Jumlah Kemasan: {count_kemasan}")
print(f"Jumlah Kapsul: {count_kapsul}")
