from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from preprocess import preprocess_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://mnist-predictor.vercel.app"}}, supports_credentials=True)

# Load model once at startup
model = tf.keras.models.load_model("mnist_cnn.h5")
print("✅ MNIST model loaded")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Save temporarily
    temp_path = "temp.png"
    file.save(temp_path)

    # Preprocess
    img = preprocess_image(temp_path)

    # Predict
    preds = model.predict(img)
    predicted_digit = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Cleanup
    os.remove(temp_path)

    return jsonify({
        "prediction": predicted_digit,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
