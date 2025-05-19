import fitz  # PyMuPDF
import os

def convert_pdf_to_image(pdf_path, output_path, dpi=150):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # Load the first page
        zoom = dpi / 72  # 72 is the default resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        
        pix.save(output_path)
        doc.close()
        return output_path
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return None


# ---------- Usage ----------

pdf_path = "check/Certificates 380-242.pdf"
image_path = "check/certificate_image.jpg"

converted_image = convert_pdf_to_image(pdf_path, image_path)

if converted_image:
    print("✅ PDF converted to image successfully.")
else:
    print("❌ Conversion failed.")
