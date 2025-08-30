import os
from PIL import Image

# Paths
PRODUCT_FOLDER = os.path.join("products", "images")
os.makedirs(PRODUCT_FOLDER, exist_ok=True)

# Loop through all images in products/images
for filename in os.listdir(PRODUCT_FOLDER):
    if filename.endswith(".png"):
        img_path = os.path.join(PRODUCT_FOLDER, filename)
        img = Image.open(img_path)
        # Convert grayscale to RGB
        img_rgb = img.convert("RGB")
        # Save back (overwrite or new file)
        img_rgb.save(img_path)

print("All images converted to RGB.")
