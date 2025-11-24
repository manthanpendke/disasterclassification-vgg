# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from pathlib import Path

st.set_page_config(page_title="Disaster Classifier", layout="centered")

# ----------------- CONFIG -----------------
# Google Drive file id (from your link)
DRIVE_FILE_ID = "1zUJt37Si2JsOFDI2d6Z1wJ9t9WtYsS5k"

# Local fallback path (you said the model exists here in your workspace)
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"

# Local filename to save after download (and loaded by Keras)
MODEL_PATH = "disaster_classifier_finetuned.keras"

# ----------------- Google Drive downloader -----------------
def get_confirm_token(resp):
    # check cookies first
    for key, value in resp.cookies.items():
        if key.startswith("download_warning") or key.startswith("consent"):
            return value
    # fallback: attempt to parse token from html
    try:
        import re
        m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
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

def download_file_from_google_drive(file_id: str, destination: str):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    resp = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(resp)
    if token:
        resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Drive download failed with status {resp.status_code}")
    save_response_content(resp, destination)
    return destination

# ----------------- Model loader (cached) -----------------
@st.cache_resource
def load_model():
    import tensorflow as tf

    # 1) If local fallback exists (e.g. in runtime or if you run locally), use it
    if os.path.exists(LOCAL_FALLBACK):
        st.info(f"Using local model at {LOCAL_FALLBACK}")
        path_to_load = LOCAL_FALLBACK
    # 2) If we've previously downloaded it in the app instance, use that
    elif os.path.exists(MODEL_PATH):
        st.info(f"Using cached model at {MODEL_PATH}")
        path_to_load = MODEL_PATH
    # 3) Otherwise download from Google Drive
    else:
        st.info("Model not found locally — downloading from Google Drive (first run only)...")
        try:
            download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
            path_to_load = MODEL_PATH
        except Exception as e:
            st.error(f"Model download failed: {e}")
            raise

    # Load Keras model (compile=False speeds loading)
    model = tf.keras.models.load_model(path_to_load, compile=False)
    return model

# ----------------- UI -----------------
st.title("📸 Disaster Classifier (Keras)")
st.write("Uploads an image, downloads/loads the model (first run), and displays prediction.")

with st.spinner("Loading model (cached) — this may take time on first run..."):
    model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to run inference.")
    st.stop()

# display image
img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input image", use_column_width=True)

# ----------------- Preprocessing helper -----------------
def infer_target_size(keras_model):
    try:
        # try to read input shape; supports common formats
        shape = None
        if hasattr(keras_model, "inputs") and keras_model.inputs:
            shape = keras_model.inputs[0].shape
        elif hasattr(keras_model, "input_shape"):
            shape = keras_model.input_shape
        if shape is not None:
            # shape may be (None, H, W, C) or (None, C, H, W)
            dims = [int(s) for s in shape if s is not None and s != 0]
            if len(dims) >= 2:
                # take last two as H, W or H, W from dims
                if len(dims) >= 3:
                    # assume dims = H, W, C
                    h, w = dims[0], dims[1]
                else:
                    h, w = dims[0], dims[1]
                # sanity check
                if 8 <= h <= 2048 and 8 <= w <= 2048:
                    return (w, h)
    except Exception:
        pass
    return (224, 224)

TARGET_SIZE = infer_target_size(model)
st.write(f"Using input size: {TARGET_SIZE[0]} x {TARGET_SIZE[1]} (width x height)")

# Preprocess image
img_resized = img.resize((TARGET_SIZE[0], TARGET_SIZE[1]))
arr = np.array(img_resized).astype(np.float32) / 255.0

# ensure 3 channels
if arr.ndim == 2:
    arr = np.stack([arr] * 3, axis=-1)
if arr.shape[-1] == 4:
    arr = arr[..., :3]

batch = np.expand_dims(arr, axis=0)

# predict
with st.spinner("Running model inference..."):
    preds = model.predict(batch)

st.subheader("Raw model output (numpy array)")
st.write(preds)

# display top prediction (works for classification)
try:
    top_idx = int(np.argmax(preds, axis=1)[0])
    top_prob = float(np.max(preds, axis=1)[0])
    label_name = None
    if Path("labels.txt").exists():
        with open("labels.txt", "r") as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]
        if 0 <= top_idx < len(labels):
            label_name = labels[top_idx]
    if label_name:
        st.success(f"Predicted: **{label_name}**  (score: {top_prob:.3f})")
    else:
        st.success(f"Predicted class index: **{top_idx}**  (score: {top_prob:.3f})")
except Exception:
    st.info("Prediction output couldn't be interpreted as classification scores. Check model output shape.")
