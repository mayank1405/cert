import fitz  # PyMuPDF
import cv2
import numpy as np
import tensorflow as tf
import os

folder_path="C:\\Users\\MAYANK\\Desktop\\certificate_validation\\check"
files_for_deletion=['certificate_image.jpg']
# Paths
MODEL_PATH = "models/certificate_cropped_deepcnn.h5"
# Constants
IMG_SIZE = (224, 224)
CROP_COORDS = {
    "x_start_ratio": 0.78,
    "x_end_ratio": 0.97,
    "y_start_ratio": 0.20,
    "y_end_ratio": 0.55
}
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")
# Function to convert PDF to Image using PyMuPDF
def convert_pdf_to_image(pdf_path, output_path, dpi=150):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # Load the first page
        zoom = dpi / 72  # 72 is the default resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        pix.save(output_path)
        doc.close()
        
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return None

# Function to crop and preprocess the image
def crop_and_preprocess(image_path):
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

    # Resize and normalize
    resized = cv2.resize(cropped, IMG_SIZE)
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)  # Shape: (1, 224, 224, 3)

    return input_image

# Full pipeline to convert, crop, and preprocess the PDF
# def process_pdf(pdf_path):
#     image = convert_pdf_to_image(pdf_path)
    # if image is not None:
    #     preprocessed_image = crop_and_preprocess(image)
    #     return preprocessed_image
    # return None

# ---------- Usage ----------

pdf_path = "check/Certificates 380-242.pdf"
image_path = "check/certificate_image.jpg"

convert_pdf_to_image(pdf_path, image_path)
input_image=crop_and_preprocess(image_path)
# preprocessed_image = process_pdf(pdf_path, image_path)




image = input_image

prediction = model.predict(image)[0][0]
label = "Authentic" if prediction < 0.5 else "Tampered"
confidence = (1 - prediction) if label == "Authentic" else prediction

# Output result
print("\n🔍 Prediction Result:")
print(f"Predicted Label: {label}")
print(f"Confidence: {confidence * 100:.2f}%")


for file in os.listdir(folder_path):
    file_path= os.path.join(folder_path, file)

    if os.path.isfile(file_path) and file in files_for_deletion:
        os.remove(file_path)




# if preprocessed_image is not None:
#     print("✅ PDF processed and preprocessed successfully.")
# else:
#     print("❌ Processing failed.")
