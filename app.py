# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import importlib
import inspect
import json
import os
import requests
from pathlib import Path
import shutil
import time

st.set_page_config(page_title="Disaster App (wrapper)", layout="centered")

# --- CONFIG: HuggingFace URL (already uploaded by you) ---
MODEL_URL = "https://huggingface.co/manthanpendke/disaster-classifier-model/resolve/main/disaster_classifier_finetuned.keras"
MODEL_LOCAL = "disaster_classifier_finetuned.keras"
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"
MIN_OK_SIZE = 5 * 1024 * 1024  # 5 MB

# --- download helper (HuggingFace direct link) ---
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

def download_from_hf(url, dst, retries=2):
    session = requests.Session()
    for attempt in range(1, retries+1):
        st.info(f"Downloading model (attempt {attempt}/{retries})...")
        r = session.get(url, stream=True, timeout=300)
        try:
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Download request failed: {e}")
        save_stream_to_file(r, dst)
        size = os.path.getsize(dst) if os.path.exists(dst) else 0
        st.write(f"Downloaded {dst} — size={size:,} bytes")
        if size < MIN_OK_SIZE or is_probably_html(dst):
            st.warning("Downloaded file invalid or too small; retrying...")
            try:
                os.remove(dst)
            except:
                pass
            time.sleep(1)
            continue
        return dst
    raise RuntimeError("Failed to download a valid model file from HuggingFace after retries.")

# --- ensure model present (local fallback -> downloaded) ---
def ensure_model():
    if os.path.exists(LOCAL_FALLBACK):
        st.info(f"Using local fallback model at {LOCAL_FALLBACK}")
        return LOCAL_FALLBACK
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) >= MIN_OK_SIZE and not is_probably_html(MODEL_LOCAL):
        st.info(f"Using cached model at {MODEL_LOCAL}")
        return MODEL_LOCAL
    # else download
    return download_from_hf(MODEL_URL, MODEL_LOCAL)

# optional helper to remove cached file
def remove_cached_model():
    if os.path.exists(MODEL_LOCAL):
        os.remove(MODEL_LOCAL)

# --- dynamic importer + function finder ---
def import_anna_module():
    # anna.py must be in repo root. Use importlib to import/reload
    try:
        import anna
        importlib.reload(anna)
    except ModuleNotFoundError:
        # Try importing via runpy if not a proper module
        raise RuntimeError("Cannot import module 'anna'. Ensure anna.py is in the repo root and has no syntax errors.")
    return anna

def find_predict_function(module):
    # try common function names and signatures
    candidates = [
        "analyze_image",
        "predict_image",
        "predict",
        "inference",
        "infer",
        "run_inference",
        "run",
        "main_predict"
    ]
    for name in candidates:
        if hasattr(module, name) and callable(getattr(module, name)):
            func = getattr(module, name)
            # check signature: prefer functions accepting 1 arg (image path)
            sig = inspect.signature(func)
            if len(sig.parameters) >= 1:
                return func
            else:
                return func
    # fallback: if module itself defines a function `get_result` etc.
    # last resort: check for a top-level 'main' that accepts args or uses global
    if hasattr(module, "main") and callable(module.main):
        return module.main
    return None

# --- result normalizer ---
def normalize_result(res):
    """
    Accept many return types:
    - dict -> return as-is
    - json string -> parse
    - tuple/list -> map by position if we can
    - numpy array -> try to interpret
    """
    if isinstance(res, dict):
        return res
    # JSON string
    if isinstance(res, str):
        try:
            return json.loads(res)
        except:
            pass
    # list/tuple
    if isinstance(res, (list, tuple)):
        # try common order: [pred_class, class_conf, severity, authority]
        try:
            pred = res[0]
            # If first element is dict-like:
            if isinstance(pred, dict):
                return pred
            out = {}
            if len(res) >= 1:
                out["predicted_class"] = res[0]
            if len(res) >= 2:
                out["class_confidence"] = float(res[1])
            if len(res) >= 3:
                out["estimated_severity"] = float(res[2])
            if len(res) >= 4:
                out["responsible_authority"] = res[3]
            return out
        except Exception:
            pass
    # numpy arrays
    try:
        import numpy as _np
        if isinstance(res, _np.ndarray):
            # treat as probabilities
            if res.ndim == 2 and res.shape[0] == 1:
                probs = res[0]
                idx = int(_np.argmax(probs))
                conf = float(_np.max(probs))
                return {"predicted_class": idx, "class_confidence": conf, "estimated_severity": conf}
    except Exception:
        pass
    # fallback
    return {"raw_output": str(res)}

# --- map severity to label ---
def severity_label(percent):
    if percent < 40:
        return "Low"
    elif percent < 80:
        return "Medium"
    else:
        return "High"

# --- UI ---
st.title("Disaster App — using anna.py backend")
st.write("Uploads an image and runs `anna.py` prediction logic (no rewrite).")

# let user optionally clear cached model
if st.button("Delete cached model file (force re-download)"):
    remove_cached_model()
    st.experimental_rerun()

# ensure model is present (download if necessary)
with st.spinner("Ensuring model availability..."):
    try:
        model_path = ensure_model()
    except Exception as e:
        st.error(f"Model ensure failed: {e}")
        st.stop()

# Import anna module and set model path if it exposes a variable or setter
with st.spinner("Importing anna.py..."):
    try:
        anna = import_anna_module()
    except Exception as e:
        st.error(f"Import failed: {e}")
        st.stop()

# If anna exposes a variable or setter to point to a model file, set it (best-effort)
# common names to try: MODEL_PATH, MODEL_FILE, model_path, set_model_path
for attr in ("MODEL_PATH", "MODEL_FILE", "model_path", "modelFile"):
    if hasattr(anna, attr):
        try:
            setattr(anna, attr, model_path)
            st.info(f"Set anna.{attr} -> {model_path}")
        except Exception:
            pass

if hasattr(anna, "set_model_path") and callable(getattr(anna, "set_model_path")):
    try:
        anna.set_model_path(model_path)
        st.info("Called anna.set_model_path(...)")
    except Exception:
        pass

# find a good prediction function
predict_fn = find_predict_function(anna)
if not predict_fn:
    st.error("Could not find a callable prediction function in anna.py. Make sure it defines one of: analyze_image, predict_image, predict, inference, infer, run_inference.")
    st.stop()

st.write(f"Using prediction function: `{predict_fn.__name__}` from anna.py")

# file uploader
uploaded = st.file_uploader("Upload image (jpg/png/jpeg)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to run inference.")
    st.stop()

# save uploaded image to temp file and call anna
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.image(Image.open(tmp_path), caption="Uploaded image", use_column_width=True)

# call the backend function
with st.spinner("Running anna prediction..."):
    try:
        # try calling with one argument (image path)
        res = None
        try:
            res = predict_fn(tmp_path)
        except TypeError:
            # try calling without args
            try:
                res = predict_fn()
            except TypeError:
                # try calling with kwargs
                try:
                    res = predict_fn(image_path=tmp_path)
                except Exception as e:
                    raise RuntimeError(f"Prediction function call failed: {e}")

        norm = normalize_result(res)
    except Exception as e:
        st.error(f"Prediction call failed: {e}")
        st.stop()

# normalize fields and show final results
predicted_class = norm.get("predicted_class") or norm.get("predicted_disaster") or norm.get("predicted_label") or norm.get("label") or norm.get("class")
class_conf = norm.get("class_confidence") or norm.get("confidence") or norm.get("class_conf") or 0.0
severity = norm.get("estimated_severity") or norm.get("severity") or norm.get("severity_percent") or norm.get("severity_score")
# If severity is in 0..1 convert to percentage
try:
    severity = float(severity)
    if 0.0 <= severity <= 1.0:
        severity_pct = severity * 100.0
    else:
        severity_pct = severity
except Exception:
    severity_pct = 0.0

responsible_authority = norm.get("responsible_authority") or norm.get("authority") or "Local Municipality / Disaster Management"

# if predicted_class is int index, try to resolve labels.txt
if isinstance(predicted_class, (int, float)) and Path("labels.txt").exists():
    try:
        with open("labels.txt") as f:
            labs = [l.strip() for l in f if l.strip()]
        idx = int(predicted_class)
        if 0 <= idx < len(labs):
            predicted_class = labs[idx]
    except Exception:
        pass

severity_lvl = severity_label(severity_pct)

# display results
st.subheader("Result")
st.write(f"**Disaster:** {predicted_class}")
st.write(f"**Responsible authority:** {responsible_authority}")
st.write(f"**Severity:** {severity_pct:.1f}% ({severity_lvl})")
st.write(f"**Confidence:** {float(class_conf):.3f}")

st.subheader("Raw normalized output (json)")
st.json(norm)

# done
