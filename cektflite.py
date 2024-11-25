import tensorflow as tf

def inspect_tflite_model(model_path):
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("=== Model Input Details ===")
        for detail in input_details:
            print(f"Name: {detail['name']}")
            print(f"Shape: {detail['shape']}")
            print(f"Data type: {detail['dtype']}")
            print("-" * 30)

        print("\n=== Model Output Details ===")
        for idx, detail in enumerate(output_details):
            print(f"Output {idx + 1}:")
            print(f"Name: {detail['name']}")
            print(f"Shape: {detail['shape']}")
            print(f"Data type: {detail['dtype']}")
            print("-" * 30)

        # Check possible bounding boxes limit
        detection_boxes_shape = output_details[1]['shape']
        max_detections = detection_boxes_shape[1]  # Second dimension usually represents max detections
        print(f"\nMaximum number of detections supported by this model: {max_detections}")

    except Exception as e:
        print(f"Error while inspecting the TFLite model: {e}")


# Path to your TFLite model
model_path = "newmodel.tflite"
inspect_tflite_model(model_path)
