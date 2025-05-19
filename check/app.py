from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import tensorflow as tf





app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
#CHANGE
# Paths
CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.normpath(os.path.join(CURRENT_PATH, "..","models", "certificate_cropped_deepcnn.h5"))
folder_path=CURRENT_PATH
# folder_path="C:\\Users\\MAYANK\\Desktop\\certificate_validation\\check"
files_for_deletion=['certificate_image.jpg','Certificates 380-242.pdf'] #'Certificates 380-242.pdf'
# MODEL_PATH = "models/certificate_cropped_deepcnn.h5"
UPLOAD_FOLDER=CURRENT_PATH
# UPLOAD_FOLDER="C:\\Users\\MAYANK\\Desktop\\certificate_validation\\check"

pdf_path = "check/Certificates 380-242.pdf"
image_path = "check/certificate_image.jpg"

# Constants
IMG_SIZE = (224, 224)
CROP_COORDS = {
    "x_start_ratio": 0.78,
    "x_end_ratio": 0.97,
    "y_start_ratio": 0.20,
    "y_end_ratio": 0.55
}
model = tf.keras.models.load_model(MODEL_PATH)

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








#ENDCHANGE
@app.route("/")
def func():
    return "<h1>HELLO!</h1>"

@app.route("/pdfpage", methods=['POST'])
def pdfpage():


    #SAVE FILE 
    if request.method=='POST':
        if 'file' not in request.files:
            return jsonify({'error':'No file part in the request'}), 400

        file=request.files['file']
        if file.filename=='':
            return jsonify({'error':'no file selected'}), 400

        file.save(os.path.join(UPLOAD_FOLDER, "Certificates 380-242.pdf"))

        convert_pdf_to_image(pdf_path, image_path)
        input_image=crop_and_preprocess(image_path)
        image = input_image

        prediction = model.predict(image)[0][0]
        label = "Authentic" if prediction < 0.5 else "Tampered"
        confidence = (1 - prediction) if label == "Authentic" else prediction
        new_conf= f"{confidence * 100:.2f}"


        for file in os.listdir(folder_path):

            file_path= os.path.join(folder_path, file)

            if os.path.isfile(file_path) and file in files_for_deletion:
                os.remove(file_path)


    
    return jsonify({'confidence': new_conf, 'label':label})
    

    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=False, port=9000)