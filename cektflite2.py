import tensorflow as tf

def check_tflite_model_max_detections(model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get output details
    output_details = interpreter.get_output_details()

    print("Output Details:")
    for i, detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data Type: {detail['dtype']}")

    # Assuming the first output is bounding box coordinates
    max_detections = output_details[0]['shape'][1] if len(output_details[0]['shape']) > 1 else 0
    print(f"\nEstimated max detections supported by the model: {max_detections}")

# Replace with the path to your TFLite model
model_path = "detect_updated.tflite"
check_tflite_model_max_detections(model_path)
