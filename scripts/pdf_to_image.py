import fitz  # PyMuPDF
import os

# Define input and output folders
pdf_folders = {
    "authentic": "dataset/authentic_pdfs",
    "tampered": "dataset/tampered_pdfs"
}
output_folder = "dataset/images"
os.makedirs(output_folder, exist_ok=True)

# Function to convert PDFs to images
def convert_pdfs_to_images(pdf_folder, category):
    save_folder = os.path.join(output_folder, category)
    os.makedirs(save_folder, exist_ok=True)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)  # Open the PDF
            
            # Convert only the first page of the PDF to an image
            pix = doc[0].get_pixmap()
            image_name = os.path.splitext(pdf_file)[0] + ".png"
            image_path = os.path.join(save_folder, image_name)
            
            pix.save(image_path)  # Save as PNG
            print(f"Converted: {pdf_file} → {image_path}")

# Convert both authentic and tampered certificates
convert_pdfs_to_images(pdf_folders["authentic"], "authentic")
convert_pdfs_to_images(pdf_folders["tampered"], "tampered")

print("✅ All PDFs converted to images successfully!")
