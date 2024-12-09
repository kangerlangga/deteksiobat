import os
from tensorflow.lite.python.interpreter import Interpreter

def check_tflite_model_details(model_path):
    """
    Memuat model TFLite dan mencetak detail input/output tensor.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model TFLite tidak ditemukan: {model_path}")

    # Load model TFLite
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Ambil detail tensor input dan output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n=== Detail Input Tensor ===")
    for i, detail in enumerate(input_details):
        print(f"Input {i}: {detail}")

    print("\n=== Detail Output Tensor ===")
    for i, detail in enumerate(output_details):
        print(f"Output {i}: {detail}")

    print("\nCatatan:")
    print("Jumlah maksimum deteksi biasanya tergantung pada dimensi 'shape' output tensor.")
    print("Misalnya, jika shape-nya [1, 10, 4], maka model hanya mendukung maksimal 10 deteksi per gambar.")

# Ganti PATH_TO_MODEL dengan path ke file model TFLite Anda
PATH_TO_MODEL = 'dataset/detect.tflite'

# Jalankan fungsi untuk memeriksa model
check_tflite_model_details(PATH_TO_MODEL)
