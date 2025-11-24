# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from pathlib import Path

st.set_page_config(page_title="Disaster Classifier", layout="centered")

# ---------------- Google Drive downloader (handles large-file confirmation) ----------------
DRIVE_FILE_ID = "1zUJt37Si2JsOFDI2d6Z1wJ9t9WtYsS5k"
MODEL_PATH = "disaster_classifier_finetuned.keras"   # local filename to save as

def get_confirm_token(resp):
    # try cookies first (common)
    for key, value in resp.cookies.items():
        if key.startswith("download_warning") or key.startswith("consent"):
            return value
    # fallback: search response text for confirm token
    try:
        text = resp.text
        import re
        m = re.search(r"confirm=([0-9A-Za-z_]+)", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    resp = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(resp)

    if token:
        resp = session.get(URL, params={"id": id, "confirm": token}, stream=True)

    if resp.status_code != 200:
        raise ValueError(f"Failed to start download (status {resp.status_code})")

    save_response_content(resp, destination)
    return destination

# ---------------- model loader (cached) ----------------
@st.cache_resource
def load_model_from_drive():
    import tensorflow as tf

    # If file already exists on instance, skip download
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally — downloading from Google Drive (first run only).")
        try:
            download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
        except Exception as e:
            st.error(f"Model download failed: {e}")
            raise

    # Load keras model (compile=False to speed up)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# ---------------- UI ----------------
st.title("📸 Disaster Classification (Keras model)")
st.write("Uploads an image, downloads the model from Google Drive (first run), and shows prediction.")

with st.spinner("Loading model (cached) — this may take a moment on first run..."):
    model = load_model_from_drive()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to get prediction.")
    st.stop()

# show image
img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input image", use_column_width=True)

# ---------- Preprocess: adapt if your model used a different size ----------
# Default uses 224x224; change if your model expects another size
TARGET_SIZE = (224, 224)

img_resized = img.resize(TARGET_SIZE)
arr = np.array(img_resized).astype(np.float32) / 255.0
if arr.ndim == 2:  # gray -> convert to 3-channel
    arr = np.stack([arr] * 3, axis=-1)
batch = np.expand_dims(arr, axis=0)

# predict
with st.spinner("Running model inference..."):
    preds = model.predict(batch)

st.subheader("Raw model output")
st.write(preds)

# Attempt to show best class index and optional labels.txt mapping
top_idx = int(np.argmax(preds, axis=1)[0])
top_prob = float(np.max(preds, axis=1)[0])

label_name = None
if Path("labels.txt").exists():
    with open("labels.txt", "r") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    if top_idx < len(labels):
        label_name = labels[top_idx]

if label_name:
    st.success(f"Predicted: **{label_name}**  (score: {top_prob:.3f})")
else:
    st.success(f"Predicted class index: **{top_idx}**  (score: {top_prob:.3f})")

st.caption("If predictions look off, adjust TARGET_SIZE to the image size used in training and/or provide a labels.txt file mapping class indices to names.")
