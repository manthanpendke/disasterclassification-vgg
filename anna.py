# anna.py
"""
Clean inference module.
Exposes: analyze_disaster_image(image_path) -> dict
- Uses TensorFlow classifier (downloads from HF if needed)
- Uses local ultralytics YOLO (if installed and yolov8n.pt present)
  OR calls external YOLO API if YOLO_API_URL env var is set.
- Pure Pillow + NumPy for image ops (no cv2)
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageOps
import numpy as np

# --- CONFIG: change if necessary ---
HF_MODEL_URL = "https://huggingface.co/manthanpendke/disaster-classifier-model/resolve/main/disaster_classifier_finetuned.keras"
MODEL_LOCAL = "disaster_classifier_finetuned.keras"
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"  # if running locally with model uploaded
YOLO_LOCAL = "yolov8n.pt"  # if you keep this file in repo root, local YOLO will be used when available
_MIN_OK_SIZE = 5 * 1024 * 1024

# External YOLO API (optional). If set in environment, anna will call it:
YOLO_API_URL = os.environ.get("YOLO_API_URL", "").strip()
YOLO_API_KEY = os.environ.get("YOLO_API_KEY", "").strip()

# --- lazy loaded globals ---
_classifier = None
_yolo = None
_IMG_SIZE = (224, 224)
_idx_to_label = None

# -------------------- utilities --------------------
def _download_from_hf(url: str, dst_path: str, min_ok=_MIN_OK_SIZE):
    import requests
    session = requests.Session()
    r = session.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)
    size = os.path.getsize(dst_path)
    if size < min_ok:
        raise RuntimeError(f"Downloaded model too small ({size} bytes).")
    time.sleep(0.1)
    return dst_path

def _ensure_model_file():
    # prefer explicit fallback for dev
    if os.path.exists(LOCAL_FALLBACK):
        return LOCAL_FALLBACK
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) >= _MIN_OK_SIZE:
        return MODEL_LOCAL
    # download from HF
    _download_from_hf(HF_MODEL_URL, MODEL_LOCAL)
    return MODEL_LOCAL

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr

def _resize_for_model(pil: Image.Image, size):
    return pil.resize(size, Image.BILINEAR)

# -------------------- classifier loader --------------------
def _load_classifier():
    global _classifier, _IMG_SIZE
    if _classifier is not None:
        return _classifier
    import tensorflow as tf
    model_path = _ensure_model_file()
    model = tf.keras.models.load_model(model_path, compile=False)
    _classifier = model
    # try infer input size
    try:
        shape = model.inputs[0].shape
        h = int(shape[1]) if shape[1] is not None else None
        w = int(shape[2]) if shape[2] is not None else None
        if h and w:
            _IMG_SIZE = (w, h)
    except Exception:
        _IMG_SIZE = (224, 224)
    return _classifier

# -------------------- YOLO: local loader or external API --------------------
def _load_local_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    if not os.path.exists(YOLO_LOCAL):
        return None
    _yolo = YOLO(YOLO_LOCAL)
    return _yolo

def _call_external_yolo_api(image_path: str, api_url: str = YOLO_API_URL, api_key: str = YOLO_API_KEY, timeout=30) -> Optional[Dict[str, Any]]:
    if not api_url:
        return None
    import requests
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    with open(image_path, "rb") as f:
        files = {"file": f}
        try:
            r = requests.post(api_url, files=files, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

# -------------------- detection -> impact score --------------------
def compute_impact_from_detections(detections_json: Optional[Dict[str, Any]]) -> float:
    if not detections_json or "boxes" not in detections_json:
        return 0.0
    img_w = float(detections_json.get("width", 1))
    img_h = float(detections_json.get("height", 1))
    total_critical_area = 0.0
    critical_object_ids = [0, 2, 7]  # person, car, truck (COCO ids)
    for b in detections_json["boxes"]:
        cls = int(b.get("class_id", -1))
        if cls in critical_object_ids:
            x1, y1, x2, y2 = (float(b.get(k, 0.0)) for k in ("x1","y1","x2","y2"))
            total_critical_area += max(0.0, (x2 - x1) * (y2 - y1))
    density = total_critical_area / (img_w * img_h + 1e-9)
    return float(min(1.0, density * 10.0))

# -------------------- attention / gradcam-ish --------------------
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
    spread = float(np.sum(cam > (np.max(cam) * 0.5)) / (cam.size + 1e-9)) if np.max(cam) > 0 else 0.0
    return float((intensity * 0.5 + spread * 0.5) * 2.0)

# -------------------- chaos & cyclone heuristics (PIL+NumPy) --------------------
def _edge_density(arr_gray: np.ndarray) -> float:
    # simple diff-based edges
    gx = np.abs(np.diff(arr_gray, axis=1))
    gy = np.abs(np.diff(arr_gray, axis=0))
    # compose
    mag = np.zeros_like(arr_gray, dtype=np.float32)
    mag[:gx.shape[0], :gx.shape[1]] += gx.astype(np.float32)
    mag[:gy.shape[0], :gy.shape[1]] += gy.astype(np.float32)
    mx = mag.max() if mag.size else 0.0
    if mx <= 1e-9:
        return 0.0
    mag_norm = mag / (mx + 1e-9)
    return float(np.mean(mag_norm))

def _cyclone_eye_bonus(arr_gray: np.ndarray) -> float:
    h, w = arr_gray.shape[:2]
    cx, cy = w//2, h//2
    r = int(min(w,h) * 0.12)
    inner = arr_gray[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
    outer_r = int(min(w,h) * 0.25)
    outer = arr_gray[max(0,cy-outer_r):min(h,cy+outer_r), max(0,cx-outer_r):min(w,cx+outer_r)]
    if inner.size == 0 or outer.size == 0:
        return 0.0
    mean_inner = float(np.mean(inner))
    mean_outer = float(np.mean(outer))
    if mean_outer - mean_inner > 12.0:
        return 0.3
    return 0.0

# -------------------- main analysis function --------------------
def analyze_disaster_image(image_path: str) -> Dict[str, Any]:
    """
    Returns:
      {
        'image_path': '5.jpg',
        'predicted_class': 'earthquake',
        'class_confidence': 0.9999,
        'estimated_severity': 0.478,
        'responsible_authority': 'Local Municipality / Search & Rescue'
      }
    """
    clf = _load_classifier()
    # grad model: find a conv-like layer
    import tensorflow as tf
    last_conv = None
    for layer in reversed(clf.layers):
        if hasattr(layer, "kernel_size"):
            last_conv = layer
            break
    if last_conv is None:
        raise RuntimeError("No conv layer found in classifier")
    grad_model = tf.keras.Model(inputs=clf.inputs, outputs=[last_conv.output, clf.output])

    # load image via PIL
    pil = Image.open(image_path).convert("RGB")
    arr_gray = np.array(ImageOps.grayscale(pil)).astype(np.float32)

    # classifier prep
    target_w, target_h = _IMG_SIZE
    try:
        target_w, target_h = _IMG_SIZE
    except Exception:
        target_w, target_h = (224, 224)
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

    # YOLO: try local first, then external API
    impact_score = 0.0
    yolo_model = _load_local_yolo()
    detections_json = None
    if yolo_model is not None:
        try:
            # ultralytics accepts PIL image
            res = yolo_model(pil, verbose=False)
            if res:
                r = res[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None and boxes.data is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    boxes_out = []
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = [float(v) for v in xyxy[i][:4]]
                        boxes_out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": int(cls[i]), "score": float(conf[i])})
                    detections_json = {"width": pil.width, "height": pil.height, "boxes": boxes_out}
        except Exception:
            detections_json = None

    # external YOLO if configured and no local detections
    if detections_json is None and YOLO_API_URL:
        detections_json = _call_external_yolo_api(image_path)

    if detections_json:
        impact_score = compute_impact_from_detections(detections_json)
    else:
        impact_score = 0.0

    chaos_score = _edge_density(arr_gray)
    base_severity = (attention_score * 0.5) + (impact_score * 0.3) + (chaos_score * 0.2)

    # load labels (best-effort)
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

    # authority mapping (same logic as your notebook)
    authority_map = {
        'cyclone':    (lambda s: 'State Emergency Services (SES)' if s < 0.5 else 'National Disaster Response Force (NDRF)'),
        'earthquake': (lambda s: 'Local Municipality / Search & Rescue' if s < 0.5 else 'NDRF + State Government'),
        'flood':      (lambda s: 'Municipal Water Dept / SES' if s < 0.5 else 'State Flood Response + NDRF'),
        'wildfire':   (lambda s: 'Local Fire Brigade' if s < 0.5 else 'National Fire Services + Forest Dept')
    }
    auth_func = authority_map.get(predicted_label, (lambda s: "Local Authorities"))
    responsible_authority = auth_func(final_severity)

    return {
        'image_path': os.path.basename(image_path),
        'predicted_class': predicted_label,
        'class_confidence': round(class_conf, 4),
        'estimated_severity': round(final_severity, 4),
        'responsible_authority': responsible_authority
    }

# Convenience: allow CLI run
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        out = analyze_disaster_image(sys.argv[1])
        print(json.dumps(out, indent=2))
    else:
        print("Call analyze_disaster_image('/path/to/image.jpg')")
