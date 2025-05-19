import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Folder paths
image_folders = {
    "authentic": "cropped_dataset/authentic",
    "tampered": "cropped_dataset/tampered"
}

# Image size suitable for your custom CNN
IMG_SIZE = (224, 224)

# Data holders
X = []
y = []

# Load and preprocess images
def load_images(folder, label):
    for image_file in os.listdir(folder):
        image_path = os.path.join(folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read {image_file}")
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # Normalize
        X.append(img)
        y.append(label)

# Load images from both folders
load_images(image_folders["authentic"], label=0)
load_images(image_folders["tampered"], label=1)

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# First split into train+val and test sets (15% test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Then split temp into train and validation (15% val of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)

# Augmentation setup (for use during model training)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Save preprocessed datasets
os.makedirs("cropped_dataset", exist_ok=True)
np.save("cropped_dataset/X_train.npy", X_train)
np.save("cropped_dataset/X_val.npy", X_val)
np.save("cropped_dataset/X_test.npy", X_test)
np.save("cropped_dataset/y_train.npy", y_train)
np.save("cropped_dataset/y_val.npy", y_val)
np.save("cropped_dataset/y_test.npy", y_test)

# Summary
print("✅ Preprocessing and augmentation setup complete!")
print(f"Total images processed: {len(X)}")
print(f"Training images: {len(X_train)}")
print(f"Validation images: {len(X_val)}")
print(f"Test images: {len(X_test)}")
