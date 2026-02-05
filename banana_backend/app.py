from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/model.tflite"
LABELS_PATH = "model/labels.txt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model.tflite not found")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route("/")
def home():
    return jsonify({
        "message": "Banana Disease Detection API is running 🚀",
        "status": "live"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(output))

    return jsonify({
        "prediction": labels[idx],
        "confidence": round(float(output[idx]) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
