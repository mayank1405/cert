import numpy as np
import tensorflow as tf
import os

# Paths
MODEL_PATH = "models/certificate_cropped_deepcnn.h5"
PREPROCESSED_IMAGE_PATH = "check/preprocessed_image.npy"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

# Load preprocessed image
if not os.path.exists(PREPROCESSED_IMAGE_PATH):
    print("❌ Preprocessed image not found. Please run crop_and_preprocess.py first.")
    exit()

image = np.load(PREPROCESSED_IMAGE_PATH)

# Predict
prediction = model.predict(image)[0][0]
label = "Authentic" if prediction < 0.5 else "Tampered"
confidence = (1 - prediction) if label == "Authentic" else prediction

# Output result
print("\n🔍 Prediction Result:")
print(f"Predicted Label: {label}")
print(f"Confidence: {confidence * 100:.2f}%")
