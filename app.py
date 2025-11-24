# app.py (robust downloader + checker)
import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from pathlib import Path
import shutil
import time

st.set_page_config(page_title="Disaster Classifier (robust)", layout="centered")

# ---------------- CONFIG ----------------
FILE_ID = "1zUJt37Si2JsOFDI2d6Z1wJ9t9WtYsS5k"
DRIVE_URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"

# Local fallback (you said this exists in your environment)
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"

# Where we save the model in the app instance
MODEL_PATH = "disaster_classifier_finetuned.keras"

# Minimum acceptable size for the real model (bytes). Your model is ~113 MB -> use 5MB as safety lower bound.
MIN_OK_SIZE = 5 * 1024 * 1024

# ---------------- Helpers ----------------
def is_probably_html(path):
    """Check first bytes to detect HTML (Drive returned a webpage instead of binary)."""
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        # common HTML indicators
        return b"<html" in head or b"doctype html" in head or head.strip().startswith(b"<!doctype")
    except Exception:
        return False

def file_info(path):
    p = Path(path)
    return {"exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}

def remove_path(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception as e:
        st.warning(f"Could not remove {path}: {e}")

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def download_drive_file(file_id, destination, max_retries=2):
    """Robust Drive downloader that handles confirm token and retries on small/corrupt files."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    for attempt in range(1, max_retries + 1):
        st.info(f"Download attempt {attempt}/{max_retries}...")
        resp = session.get(URL, params={"id": file_id}, stream=True)
        # try cookie token
        token = None
        for key, val in resp.cookies.items():
            if key.startswith("download_warning") or key.startswith("consent"):
                token = val
                break
        if token:
            resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

        # save
        save_response_content(resp, destination)

        # quick checks
        size = os.path.getsize(destination) if os.path.exists(destination) else 0
        st.write(f"Downloaded {destination} — size={size:,} bytes")

        # detect HTML or too-small file
        if size < MIN_OK_SIZE:
            st.warning(f"Downloaded file too small ({size} bytes). Will check for HTML/corruption.")
            if is_probably_html(destination):
                st.warning("Downloaded file appears to be HTML (Google Drive returned a webpage). Deleting and retrying.")
                remove_path(destination)
                time.sleep(1)
                continue
            else:
                st.warning("Downloaded file is small but does not look like HTML. Retrying once.")
                remove_path(destination)
                time.sleep(1)
                continue
        # passed checks
        return destination

    raise RuntimeError("Failed to download a valid model file from Google Drive after retries.")

# ---------------- Model loader ----------------
@st.cache_resource
def load_model_safe():
    # lazy import here so streamlit doesn't import heavy libs too early
    import tensorflow as tf

    # 1) If the file exists at the LOCAL_FALLBACK path, prefer it
    if os.path.exists(LOCAL_FALLBACK):
        st.success(f"Using local fallback model at: {LOCAL_FALLBACK}")
        path_to_load = LOCAL_FALLBACK
    # 2) If we've already downloaded it inside this instance, and it looks OK -> use it
    elif os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_OK_SIZE and not is_probably_html(MODEL_PATH):
        st.success(f"Using previously downloaded model at: {MODEL_PATH}")
        path_to_load = MODEL_PATH
    # 3) Otherwise download from Drive
    else:
        st.info("Model not found locally — will download from Google Drive now (first run only).")
        # ensure any old corrupted file is removed
        if os.path.exists(MODEL_PATH):
            st.warning("Removing previously downloaded (possibly corrupted) file and retrying download.")
            remove_path(MODEL_PATH)
        download_drive_file(FILE_ID, MODEL_PATH)
        path_to_load = MODEL_PATH

    # Final file checks before loading
    info = file_info(path_to_load)
    st.write(f"Model file info: exists={info['exists']}, size={info['size']:,} bytes")
    if info["size"] < MIN_OK_SIZE:
        raise RuntimeError(f"Model file exists but is too small ({info['size']} bytes). Aborting load.")

    if is_probably_html(path_to_load):
        raise RuntimeError("Model file appears to be HTML (Drive returned a webpage). Delete and retry with a valid link.")

    # Try to load model and provide transparent errors
    try:
        st.info("Loading Keras model (this may take a while)...")
        model = tf.keras.models.load_model(path_to_load, compile=False)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        # provide file debug info to help root-cause (size + first bytes)
        head = None
        try:
            with open(path_to_load, "rb") as f:
                head = f.read(512)
        except Exception:
            head = b"<could not read head>"
        err_msg = (
            f"Failed to load model from {path_to_load}. Exception: {e}\n\n"
            f"File size: {info['size']} bytes\n"
            f"First 512 bytes (hex/utf8 preview): {head[:200]!r}\n"
            "If the preview looks like HTML (starts with '<' or contains 'DOCTYPE'/'html'),"
            " Google Drive returned a webpage instead of the binary model file."
        )
        # raise a new error so it's visible in Streamlit logs
        raise RuntimeError(err_msg)

# ---------------- UI ----------------
st.title("📸 Disaster Classifier (robust loader)")

try:
    with st.spinner("Preparing model (cached) — may take a minute on first run..."):
        model = load_model_safe()
except Exception as e:
    st.error("Model loading failed. See details below.")
    st.code(str(e))
    # show quick actions
    if st.button("Delete cached model file and retry"):
        if os.path.exists(MODEL_PATH):
            remove_path(MODEL_PATH)
            st.experimental_rerun()
        else:
            st.info("No cached model at expected path.")
    if os.path.exists(LOCAL_FALLBACK):
        st.info(f"Local fallback exists at {LOCAL_FALLBACK}. If you want to use it, ensure permissions are correct.")
    st.stop()

# upload and inference UI
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an image to run inference.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, use_column_width=True)

# infer input size if possible (safe fallback to 224x224)
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
                return (w, h)
    except Exception:
        pass
    return (224, 224)

target = infer_size(model)
st.write(f"Using input size: {target[0]} x {target[1]} (w x h)")

img_resized = img.resize((target[0], target[1]))
arr = np.array(img_resized).astype(np.float32) / 255.0
if arr.ndim == 2:
    arr = np.stack([arr]*3, axis=-1)
if arr.shape[-1] == 4:
    arr = arr[..., :3]
batch = np.expand_dims(arr, axis=0)

with st.spinner("Running inference..."):
    preds = model.predict(batch)

st.subheader("Model output (raw)")
st.write(preds)

try:
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds, axis=1)[0])
    label = None
    if Path("labels.txt").exists():
        with open("labels.txt") as f:
            labels = [l.strip() for l in f if l.strip()]
        if 0 <= idx < len(labels):
            label = labels[idx]
    if label:
        st.success(f"Predicted: {label} (score {prob:.3f})")
    else:
        st.success(f"Predicted class index: {idx} (score {prob:.3f})")
except Exception:
    st.info("Couldn't interpret output as classification probabilities. Inspect raw `preds` above.")
