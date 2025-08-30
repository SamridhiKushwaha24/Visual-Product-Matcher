import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCT_FOLDER = os.path.join(BASE_DIR, "products", "images")
METADATA_FILE = os.path.join(BASE_DIR, "products", "metadata.csv")

# ---------------------------
# Load metadata
# ---------------------------
if os.path.exists(METADATA_FILE):
    metadata = pd.read_csv(METADATA_FILE)
else:
    st.error(f"Metadata file not found: {METADATA_FILE}")
    st.stop()

# ---------------------------
# Load pre-trained model for embeddings
# ---------------------------
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# ---------------------------
# Function to get image embeddings
# ---------------------------
def get_embedding(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    emb = model.predict(img_array)
    return emb

# ---------------------------
# Precompute embeddings for product images
# ---------------------------
if 'product_embeddings' not in st.session_state:
    st.session_state.product_embeddings = []
    st.session_state.product_images = []
    for idx, row in metadata.iterrows():
        img_path = os.path.join(PRODUCT_FOLDER, row['filename'])
        if os.path.exists(img_path):
            emb = get_embedding(img_path)
            st.session_state.product_embeddings.append(emb)
            st.session_state.product_images.append((img_path, row['name']))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Visual Product Matcher")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Get embedding of uploaded image
    uploaded_image.save("temp_upload.png")  # save temporarily
    upload_emb = get_embedding("temp_upload.png")
    os.remove("temp_upload.png")

    # Compute cosine similarity
    similarities = []
    for emb in st.session_state.product_embeddings:
        sim = cosine_similarity(upload_emb, emb)[0][0]
        similarities.append(sim)

    # Get top 5 most similar images
    top_idx = np.argsort(similarities)[-5:][::-1]

    st.subheader("Top 5 Similar Products")
    for i in top_idx:
        img_path, name = st.session_state.product_images[i]
        st.image(img_path, caption=name, width=300)
