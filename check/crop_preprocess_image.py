import cv2
import os
import numpy as np

# Constants
IMG_PATH = "check/certificate_image.jpg"        # Input image path
CROPPED_PATH = "check/cropped_photo.jpg"        # Output cropped image path
PREPROCESSED_NPY = "check/preprocessed_image.npy"  # Output preprocessed .npy path
IMG_SIZE = (224, 224)

# Crop region ratios
CROP_COORDS = {
    "x_start_ratio": 0.78,
    "x_end_ratio": 0.97,
    "y_start_ratio": 0.20,
    "y_end_ratio": 0.55
}

def crop_and_preprocess(image_path, cropped_path, preprocessed_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to read image.")
        return

    h, w, _ = image.shape

    # Compute crop coordinates
    x1 = int(w * CROP_COORDS["x_start_ratio"])
    x2 = int(w * CROP_COORDS["x_end_ratio"])
    y1 = int(h * CROP_COORDS["y_start_ratio"])
    y2 = int(h * CROP_COORDS["y_end_ratio"])

    cropped = image[y1:y2, x1:x2]

    # Save cropped image
    os.makedirs(os.path.dirname(cropped_path), exist_ok=True)
    cv2.imwrite(cropped_path, cropped)
    print(f"✅ Cropped image saved at {cropped_path}")

    # Resize and normalize
    resized = cv2.resize(cropped, IMG_SIZE)
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)  # Shape: (1, 224, 224, 3)

    # Save preprocessed image
    np.save(preprocessed_path, input_image)
    print(f"✅ Preprocessed image saved at {preprocessed_path}")

if __name__ == "__main__":
    crop_and_preprocess(IMG_PATH, CROPPED_PATH, PREPROCESSED_NPY)
