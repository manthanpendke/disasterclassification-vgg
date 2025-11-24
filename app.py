import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
import tensorflow as tf

st.set_page_config(page_title="Disaster Classifier", layout="centered")

# ---------------- GOOGLE DRIVE SETTINGS ----------------
FILE_ID = "1zUJt37Si2JsOFDI2d6Z1wJ9t9WtYsS5k"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "disaster_classifier_finetuned.keras"

# ---------------- FIXED GOOGLE DRIVE DOWNLOADER ----------------
def download_from_google_drive(url, destination):
    session = requests.Session()
    response = session.get(url, stream=True)

    # 1) Try to get confirm token (for large files)
    def get_token(resp):
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                return val
        return None

    token = get_token(response)

    if token:
        params = {"id": FILE_ID, "confirm": token}
        response = session.get(url, params=params, stream=True)

    # 2) Download the content safely
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    # 3) Verify file size — MUST be > 5MB (your model is 113MB)
    if os.path.getsize(destination) < 5_000_000:
        raise ValueError("Downloaded file too small → Google Drive blocked the file.")

# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model from Google Drive (first run only)...")
        download_from_google_drive(MODEL_URL, MODEL_PATH)

    st.success("Model downloaded. Loading...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# ---------------- UI ----------------
st.title("📸 Disaster Classification (Keras Model)")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to start.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, use_column_width=True)

# preprocess
img_resized = img.resize((224, 224))
arr = np.array(img_resized) / 255.0
arr = np.expand_dims(arr, 0)

# predict
preds = model.predict(arr)
st.subheader("Raw Output")
st.write(preds)

top_idx = int(np.argmax(preds))
st.success(f"Predicted class index: {top_idx}")
