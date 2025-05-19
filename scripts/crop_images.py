import cv2
import os

# Define source and target folders
source_target_pairs = [
    ("dataset/images/authentic", "cropped_dataset/authentic"),
    ("dataset/images/tampered", "cropped_dataset/tampered")
]

# Loop through each pair
for img_folder, save_folder in source_target_pairs:
    os.makedirs(save_folder, exist_ok=True)  # Create target folder if not exists

    for filename in os.listdir(img_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(img_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"❌ Can't read image: {filename}")
                continue

            h, w, _ = image.shape
            print(f"📷 Processing: {filename} | Shape: {h}x{w}")

            # Crop coordinates — adjust as per certificate layout
            x1 = int(w * 0.78)
            x2 = int(w * 0.97)
            y1 = int(h * 0.20)
            y2 = int(h * 0.55)

            cropped = image[y1:y2, x1:x2]

            output_path = os.path.join(save_folder, filename)
            cv2.imwrite(output_path, cropped)

print("✅ All Cropped Images Saved!")
