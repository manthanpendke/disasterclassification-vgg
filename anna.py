# anna.py  (OpenCV-free inference module)
# Exposes: analyze_disaster_image(image_path) -> dict
# Uses: Pillow + NumPy + tensorflow + ultralytics + requests

import os
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import tensorflow as tf

# --- CONFIG ---
HF_MODEL_URL = "https://huggingface.co/manthanpendke/disaster-classifier-model/resolve/main/disaster_classifier_finetuned.keras"
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"
MODEL_LOCAL = "disaster_classifier_finetuned.keras"
YOLO_LOCAL = "yolov8n.pt"
_MIN_OK_SIZE = 5 * 1024 * 1024

# --- helpers for download ---
def _download_from_hf(url, dst_path):
    import requests
    session = requests.Session()
    resp = session.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(32768):
            if chunk:
                f.write(chunk)
    size = os.path.getsize(dst_path)
    if size < _MIN_OK_SIZE:
        raise RuntimeError(f"Downloaded model too small ({size} bytes).")
    time.sleep(0.1)
    return dst_path

def _ensure_model_file():
    if os.path.exists(LOCAL_FALLBACK):
        return LOCAL_FALLBACK
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) >= _MIN_OK_SIZE:
        return MODEL_LOCAL
    _download_from_hf(HF_MODEL_URL, MODEL_LOCAL)
    return MODEL_LOCAL

# --- lazy loaders ---
_classifier_model = None
_yolo_model = None
_IMG_SIZE = (224, 224)
_idx_to_label = None

def _load_classifier():
    global _classifier_model, _IMG_SIZE
    if _classifier_model is not None:
        return _classifier_model
    model_path = _ensure_model_file()
    model = tf.keras.models.load_model(model_path, compile=False)
    _classifier_model = model
    try:
        shape = model.inputs[0].shape
        h = int(shape[1]) if shape[1] is not None else None
        w = int(shape[2]) if shape[2] is not None else None
        if h and w:
            _IMG_SIZE = (w, h)
    except Exception:
        _IMG_SIZE = (224, 224)
    return _classifier_model

def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics not installed. Add 'ultralytics' to requirements.") from e
    if not os.path.exists(YOLO_LOCAL):
        raise FileNotFoundError(f"{YOLO_LOCAL} not found in repo root.")
    _yolo_model = YOLO(YOLO_LOCAL)
    return _yolo_model

# --- Image utilities (PIL + NumPy) ---
def _pil_to_numpy(img: Image.Image):
    arr = np.array(img).astype(np.float32)
    # ensure RGB
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr

def _resize_for_model(img: Image.Image, size):
    return img.resize(size, Image.BILINEAR)

# simple Sobel-like gradient magnitude (numpy) for chaos/edge density
def _edge_density(arr_gray):
    # arr_gray in 0..255
    gx = np.abs(np.diff(arr_gray, axis=1))
    gy = np.abs(np.diff(arr_gray, axis=0))
    # align dims
    gx = gx[:, :-0] if gx.ndim==2 else gx
    gy = gy[:-0, :] if gy.ndim==2 else gy
    mag = np.zeros_like(arr_gray, dtype=np.float32)
    mag[:gx.shape[0], :gx.shape[1]] = gx.astype(np.float32)
    mag[:gy.shape[0], :gy.shape[1]] += gy.astype(np.float32)
    mag_norm = mag / (mag.max() + 1e-9)
    return float(np.mean(mag_norm))

# cyclone-eye heuristic: compare central circle mean to surrounding ring
def _cyclone_eye_bonus(arr_gray):
    h, w = arr_gray.shape[:2]
    cx, cy = w//2, h//2
    r = int(min(w,h) * 0.12)  # center radius
    inner = arr_gray[cy-r:cy+r, cx-r:cx+r]
    outer_r = int(min(w,h) * 0.25)
    y1, y2 = max(0, cy-outer_r), min(h, cy+outer_r)
    x1, x2 = max(0, cx-outer_r), min(w, cx+outer_r)
    outer = arr_gray[y1:y2, x1:x2]
    if inner.size == 0 or outer.size == 0:
        return 0.0
    mean_inner = float(np.mean(inner))
    mean_outer = float(np.mean(outer))
    # eye exists if center is noticeably darker than surroundings
    if mean_outer - mean_inner > 12.0:  # threshold
        return 0.3
    return 0.0

# --- YOLO impact score with PIL (ultralytics accepts PIL)
def _get_yolo_impact_score(yolo_model, pil_img):
    try:
        results = yolo_model(pil_img)  # ultralytics v8: model(pil_img)
    except Exception:
        # fallback to predict API
        results = yolo_model.predict(source=pil_img, imgsz=640)
    if not results:
        return 0.0
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.data is None:
        return 0.0
    data = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else boxes.data.cpu().numpy()
    classes = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else boxes.data.cpu().numpy()[:, -1]
    # critical object ids
    critical = [0,2,7]
    img_w, img_h = pil_img.size
    total_critical_area = 0.0
    for i, box in enumerate(data):
        x1, y1, x2, y2 = box[:4]
        cls_id = int(classes[i])
        if cls_id in critical:
            total_critical_area += max(0, (x2-x1)*(y2-y1))
    density = total_critical_area / (img_w * img_h + 1e-9)
    score = min(1.0, density * 10.0)
    return float(score)

# --- Attention / gradcam like score (TensorFlow) ---
def _get_attention_score(grad_model, img_array, class_idx):
    import tensorflow as tf
    with tf.GradientTape() as tape:
        last_conv, preds = grad_model(img_array)
        class_channel = preds[:, class_idx]
    grads = tape.gradient(class_channel, last_conv)
    if grads is None:
        return 0.0
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    cam = tf.squeeze(last_conv[0] @ pooled_grads[..., tf.newaxis])
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-9)
    cam = cam.numpy()
    intensity = float(np.mean(cam))
    spread = float(np.sum(cam > (np.max(cam)*0.5)) / (cam.size + 1e-9)) if np.max(cam) > 0 else 0.0
    return float((intensity*0.5 + spread*0.5) * 2.0)

# --- main exposed function ---
def analyze_disaster_image(image_path):
    """
    Input: path to image
    Output: dict with keys:
      image_path, predicted_class, class_confidence, estimated_severity (0..1), responsible_authority
    """
    clf = _load_classifier()
    yolo = _load_yolo()

    # build grad model: last conv + preds
    last_conv = None
    for layer in reversed(clf.layers):
        # pick first conv-like layer
        if hasattr(layer, "kernel_size"):
            last_conv = layer
            break
    if last_conv is None:
        raise RuntimeError("Could not find conv layer in classifier.")

    grad_model = tf.keras.Model(inputs=clf.inputs, outputs=[last_conv.output, clf.output])

    # load image via PIL
    pil = Image.open(image_path).convert("RGB")
    # create numpy arrays for image processing
    arr_full = _pil_to_numpy(pil)  # H,W,3 float32
    # grayscale for chaos/cyclone tests
    arr_gray = np.array(ImageOps.grayscale(pil)).astype(np.float32)

    # prep for classifier
    target_w, target_h = _IMG_SIZE
    pil_resized = _resize_for_model(pil, (target_w, target_h))
    arr_model = _pil_to_numpy(pil_resized) / 255.0
    batch = np.expand_dims(arr_model, axis=0)

    # predict
    preds = clf.predict(batch, verbose=0)
    probs = preds[0]
    class_idx = int(np.argmax(probs))
    class_conf = float(probs[class_idx])

    # attention score
    attention_score = _get_attention_score(grad_model, batch, class_idx)

    # yolo impact score (use PIL original)
    impact_score = _get_yolo_impact_score(yolo, pil)

    # chaos / edge density
    chaos_score = _edge_density(arr_gray)

    base_severity = (attention_score * 0.5) + (impact_score * 0.3) + (chaos_score * 0.2)

    # cyclone eye bonus using center vs ring heuristic
    # resolve class label mapping
    global _idx_to_label
    if _idx_to_label is None:
        labf = Path("labels.txt")
        if labf.exists():
            with open(labf, "r") as f:
                _idx_to_label = [l.strip() for l in f.readlines() if l.strip()]
        else:
            _idx_to_label = ["earthquake", "flood", "cyclone", "wildfire"]
    predicted_label = _idx_to_label[class_idx] if 0 <= class_idx < len(_idx_to_label) else f"class_{class_idx}"

    if predicted_label == "cyclone":
        base_severity += _cyclone_eye_bonus(arr_gray)

    final_severity = float(min(1.0, base_severity))

    # same authority mapping as notebook
    authority_map = {
        'cyclone':    (lambda s: 'State Emergency Services (SES)' if s < 0.5 else 'National Disaster Response Force (NDRF)'),
        'earthquake': (lambda s: 'Local Municipality / Search & Rescue' if s < 0.5 else 'NDRF + State Government'),
        'flood':      (lambda s: 'Municipal Water Dept / SES' if s < 0.5 else 'State Flood Response + NDRF'),
        'wildfire':   (lambda s: 'Local Fire Brigade' if s < 0.5 else 'National Fire Services + Forest Dept')
    }
    auth_func = authority_map.get(predicted_label, (lambda s: "Local Authorities"))
    responsible_authority = auth_func(final_severity)

    result = {
        'image_path': os.path.basename(image_path),
        'predicted_class': predicted_label,
        'class_confidence': round(class_conf, 4),
        'estimated_severity': round(final_severity, 4),
        'responsible_authority': responsible_authority
    }
    return result

# CLI convenience
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        print(json.dumps(analyze_disaster_image(sys.argv[1]), indent=2))
    else:
        print("Use analyze_disaster_image('/path/to/image.jpg')")
