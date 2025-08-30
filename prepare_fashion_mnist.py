import os
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image

# ---------------------------
# Folder setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCT_FOLDER = os.path.join(BASE_DIR, "products", "images")
os.makedirs(PRODUCT_FOLDER, exist_ok=True)

# ---------------------------
# Load Fashion MNIST dataset
# ---------------------------
(x_train, y_train), (_, _) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ---------------------------
# Save first 50 images
# ---------------------------
metadata = []
for i in range(50):
    img_array = x_train[i]
    img = Image.fromarray(img_array)
    filename = f"img_{i}.png"
    img.save(os.path.join(PRODUCT_FOLDER, filename))
    metadata.append({
        "filename": filename,
        "name": class_names[y_train[i]],
        "category": "Fashion"
    })

# ---------------------------
# Save metadata.csv
# ---------------------------
df = pd.DataFrame(metadata)
os.makedirs(os.path.join(BASE_DIR, "products"), exist_ok=True)
df.to_csv(os.path.join(BASE_DIR, "products", "metadata.csv"), index=False)

print("Sample product data created in 'products/' folder.")
