# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from pathlib import Path
import shutil
import time

st.set_page_config(page_title="Disaster Classifier (HuggingFace)", layout="centered")

# ------------- CONFIG -------------
# HuggingFace raw URL (you provided this)
MODEL_URL = "https://huggingface.co/manthanpendke/disaster-classifier-model/resolve/main/disaster_classifier_finetuned.keras"

# Local fallback path (useful when running locally in your environment)
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"

# Where the file will be saved in the Streamlit instance
MODEL_PATH = "disaster_classifier_finetuned.keras"

# Minimum acceptable size for a valid model (bytes) — your model ≈113MB, keep lower bound safe
MIN_OK_SIZE = 5 * 1024 * 1024  # 5 MB

# ------------- HELPERS -------------
def save_stream_to_file(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def is_probably_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        return b"<html" in head or b"doctype html" in head or head.strip().startswith(b"<!doctype")
    except Exception:
        return False

def remove_path(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception as e:
        st.warning(f"Could not remove {path}: {e}")

def download_from_hf(url, destination, max_retries=2):
    """Download from HuggingFace (straightforward binary download). Retries on too-small file."""
    session = requests.Session()
    for attempt in range(1, max_retries + 1):
        st.info(f"Downloading model (attempt {attempt}/{max_retries})...")
        resp = session.get(url, stream=True, timeout=300)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Download request failed: {e}")

        save_stream_to_file(resp, destination)

        size = os.path.getsize(destination) if os.path.exists(destination) else 0
        st.write(f"Downloaded {destination} — size={size:,} bytes")

        if size < MIN_OK_SIZE:
            st.warning(f"Downloaded file too small ({size} bytes). Retrying...")
            remove_path(destination)
            time.sleep(1)
            continue

        # quick html check
        if is_probably_html(destination):
            remove_path(destination)
            st.warning("Downloaded file appears to be HTML (unexpected). Retrying...")
            time.sleep(1)
            continue

        # good file
        return destination

    raise RuntimeError("Failed to download a valid model file after retries (HuggingFace).")

# ------------- MODEL LOADER (cached) -------------
@st.cache_resource
def load_model_safe():
    import tensorflow as tf

    # Prefer explicit local fallback if present
    if os.path.exists(LOCAL_FALLBACK):
        st.success(f"Using local model at {LOCAL_FALLBACK}")
        path_to_load = LOCAL_FALLBACK
    elif os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_OK_SIZE and not is_probably_html(MODEL_PATH):
        st.success(f"Using cached model at {MODEL_PATH}")
        path_to_load = MODEL_PATH
    else:
        # download from HF
        if os.path.exists(MODEL_PATH):
            st.warning("Removing previous cached file (may have been corrupted) before re-download.")
            remove_path(MODEL_PATH)
        try:
            download_from_hf(MODEL_URL, MODEL_PATH)
            path_to_load = MODEL_PATH
        except Exception as e:
            raise RuntimeError(f"Model download failed: {e}")

    # Final sanity checks
    size = os.path.getsize(path_to_load) if os.path.exists(path_to_load) else 0
    st.write(f"Model file: {path_to_load} (size={size:,} bytes)")
    if size < MIN_OK_SIZE:
        raise RuntimeError(f"Model file exists but is too small ({size} bytes). Aborting load.")
    if is_probably_html(path_to_load):
        raise RuntimeError("Model file appears to be HTML (unexpected). Aborting load.")

    # Load model
    try:
        st.info("Loading Keras model (this may take a while)...")
        model = tf.keras.models.load_model(path_to_load, compile=False)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        # include file head for diagnostics
        head = b""
        try:
            with open(path_to_load, "rb") as f:
                head = f.read(512)
        except Exception:
            head = b"<could not read head>"
        raise RuntimeError(f"Failed to load model: {e}\nFile size: {size} bytes\nFirst bytes: {head[:200]!r}")

# ------------- APP UI & INFERENCE -------------
st.title("📸 Disaster Classifier (Hugging Face hosted model)")
st.write("Model is downloaded from Hugging Face (first run) and cached in the app instance.")

# Load model (cached across reruns)
try:
    with st.spinner("Preparing model (cached) — this can take a minute on first run..."):
        model = load_model_safe()
except Exception as e:
    st.error("Model preparation failed. See details below.")
    st.code(str(e))
    if st.button("Delete cached model file and retry"):
        if os.path.exists(MODEL_PATH):
            remove_path(MODEL_PATH)
            st.experimental_rerun()
        else:
            st.info("No cached model file present.")
    st.stop()

uploaded = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to run inference.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input image", use_column_width=True)

# Try to infer model input size, fallback to 224x224
def infer_size(kmodel):
    try:
        shape = None
        if hasattr(kmodel, "inputs") and kmodel.inputs:
            shape = kmodel.inputs[0].shape
        elif hasattr(kmodel, "input_shape"):
            shape = kmodel.input_shape
        if shape:
            dims = [int(s) for s in shape if s is not None and s != 0]
            if len(dims) >= 2:
                h, w = dims[0], dims[1]
                # return (width, height)
                return (w, h)
    except Exception:
        pass
    return (224, 224)

target = infer_size(model)
st.write(f"Using input size: {target[0]} x {target[1]} (w x h)")

# Preprocess
img_resized = img.resize((target[0], target[1]))
arr = np.array(img_resized).astype(np.float32) / 255.0
if arr.ndim == 2:
    arr = np.stack([arr] * 3, axis=-1)
if arr.shape[-1] == 4:
    arr = arr[..., :3]
batch = np.expand_dims(arr, axis=0)

with st.spinner("Running inference..."):
    preds = model.predict(batch)

st.subheader("Raw model output")
st.write(preds)

# Show top class if possible
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
    st.info("Couldn't interpret predictions as classification probabilities. See raw output above.")
