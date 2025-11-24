# anna.py
"""
Inference module with correct preprocessing for common Keras image models.
Exposes: analyze_disaster_image(image_path) -> dict
- Loads TF classifier .keras from HF if needed
- Uses model-specific preprocess_input when available (VGG/ResNet/MobileNet)
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

# --- CONFIG ---
HF_MODEL_URL = "https://huggingface.co/manthanpendke/disaster-classifier-model/resolve/main/disaster_classifier_finetuned.keras"
MODEL_LOCAL = "disaster_classifier_finetuned.keras"
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"
YOLO_LOCAL = "yolov8n.pt"
_MIN_OK_SIZE = 5 * 1024 * 1024

YOLO_API_URL = os.environ.get("YOLO_API_URL", "").strip()
YOLO_API_KEY = os.environ.get("YOLO_API_KEY", "").strip()

# lazy globals
_classifier = None
_yolo = None
_IMG_SIZE = (224, 224)
_idx_to_label = None
_preprocess_fn = None

# ---------------- utilities ----------------
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
    if os.path.exists(LOCAL_FALLBACK):
        return LOCAL_FALLBACK
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) >= _MIN_OK_SIZE:
        return MODEL_LOCAL
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

# ---------------- classifier loader ----------------
def _guess_preprocess_fn(model):
    """
    Try to find a suitable tf.keras.applications preprocess_input function
    by checking popular model families. Returns function or None.
    """
    try:
        # import here to avoid global tf import at module load
        import tensorflow as tf
        from importlib import import_module
        # model.name might contain 'vgg', 'resnet', 'mobilenet', etc.
        name = getattr(model, 'name', '') or ''
        name = name.lower()
        # first try to match model.name
        if 'vgg' in name:
            mod = import_module("tensorflow.keras.applications.vgg16")
            return mod.preprocess_input
        if 'resnet' in name:
            mod = import_module("tensorflow.keras.applications.resnet50")
            return mod.preprocess_input
        if 'mobilenet' in name:
            mod = import_module("tensorflow.keras.applications.mobilenet_v2")
            return mod.preprocess_input
        if 'efficientnet' in name:
            mod = import_module("tensorflow.keras.applications.efficientnet")
            return mod.preprocess_input
    except Exception:
        pass
    # As a last attempt, try importing a few preprocess functions and test shape
    try:
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
        return vgg_pre
    except Exception:
        pass
    return None

def _load_classifier():
    global _classifier, _IMG_SIZE, _preprocess_fn
    if _classifier is not None:
        return _classifier
    import tensorflow as tf
    model_path = _ensure_model_file()
    model = tf.keras.models.load_model(model_path, compile=False)
    _classifier = model
    # infer input size
    try:
        shape = model.inputs[0].shape
        h = int(shape[1]) if shape[1] is not None else None
        w = int(shape[2]) if shape[2] is not None else None
        if h and w:
            _IMG_SIZE = (w, h)
    except Exception:
        _IMG_SIZE = (224, 224)
    # detect preprocess function (try to guess)
    _preprocess_fn = _guess_preprocess_fn(model)
    return _classifier

# ---------------- YOLO local / external ----------------
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

# ---------------- detection -> impact score ----------------
def compute_impact_from_detections(detections_json: Optional[Dict[str, Any]]) -> float:
    if not detections_json or "boxes" not in detections_json:
        return 0.0
    img_w = float(detections_json.get("width", 1))
    img_h = float(detections_json.get("height", 1))
    total_critical_area = 0.0
    critical_object_ids = [0, 2, 7]
    for b in detections_json["boxes"]:
        cls = int(b.get("class_id", -1))
        if cls in critical_object_ids:
            x1, y1, x2, y2 = (float(b.get(k, 0.0)) for k in ("x1","y1","x2","y2"))
            total_critical_area += max(0.0, (x2 - x1) * (y2 - y1))
    density = total_critical_area / (img_w * img_h + 1e-9)
    return float(min(1.0, density * 10.0))

# ---------------- attention / gradcam ----------------
def _get_attention_score(grad_model, img_array, class_idx):
    import tensorflow as tf
    # ensure tensor
    if not tf.is_tensor(img_array):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        last_conv_out, preds_raw = grad_model(img_array)
        preds_tensor = preds_raw
        if isinstance(preds_raw, (list, tuple)):
            preds_tensor = preds_raw[-1]
        if not tf.is_tensor(preds_tensor):
            preds_tensor = tf.convert_to_tensor(preds_tensor, dtype=tf.float32)
        # index class channel
        try:
            class_channel = preds_tensor[:, class_idx]
        except Exception:
            preds_squeezed = tf.squeeze(preds_tensor)
            class_channel = preds_squeezed[:, class_idx]
    grads = tape.gradient(class_channel, last_conv_out)
    if grads is None:
        return 0.0
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    weighted_map = tf.tensordot(last_conv_out[0], pooled_grads, axes=[[2],[0]])
    cam = tf.maximum(weighted_map, 0.0)
    cam_max = tf.reduce_max(cam)
    cam = cam / (cam_max + 1e-9)
    cam_np = cam.numpy()
    intensity = float(np.mean(cam_np))
    spread = float(np.sum(cam_np > (np.max(cam_np) * 0.5)) / (cam_np.size + 1e-9)) if np.max(cam_np) > 0 else 0.0
    score = float((intensity * 0.5 + spread * 0.5) * 2.0)
    if not np.isfinite(score):
        return 0.0
    return float(max(0.0, min(1.0, score)))

# ---------------- chaos & cyclone heuristics ----------------
def _edge_density(arr_gray: np.ndarray) -> float:
    gx = np.abs(np.diff(arr_gray, axis=1))
    gy = np.abs(np.diff(arr_gray, axis=0))
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

# ---------------- main analysis ----------------
def analyze_disaster_image(image_path: str) -> Dict[str, Any]:
    clf = _load_classifier()
    import tensorflow as tf
    last_conv = None
    for layer in reversed(clf.layers):
        if hasattr(layer, "kernel_size"):
            last_conv = layer
            break
    if last_conv is None:
        raise RuntimeError("No conv layer found in classifier")
    grad_model = tf.keras.Model(inputs=clf.inputs, outputs=[last_conv.output, clf.output])

    pil = Image.open(image_path).convert("RGB")
    arr_gray = np.array(ImageOps.grayscale(pil)).astype(np.float32)

    target_w, target_h = _IMG_SIZE
    pil_resized = _resize_for_model(pil, (target_w, target_h))
    arr_model = _pil_to_numpy(pil_resized)

    # APPLY MODEL-SPECIFIC PREPROCESSING IF AVAILABLE
    global _preprocess_fn
    if _preprocess_fn is None:
        # attempt lazy init if not present
        try:
            _ = _load_classifier()
        except Exception:
            pass
    if _preprocess_fn:
        try:
            # preprocess expects shape (H,W,3) and returns same shaped array
            arr_pre = _preprocess_fn(arr_model.copy())
            arr_pre = np.array(arr_pre).astype(np.float32)
        except Exception:
            arr_pre = arr_model / 255.0
    else:
        arr_pre = arr_model / 255.0

    batch = np.expand_dims(arr_pre, axis=0)

    # prediction (handle tuple/list returns robustly)
    preds = clf.predict(batch, verbose=0)
    if isinstance(preds, (list, tuple)):
        candidate = preds[-1]
        try:
            import tensorflow as _tf
            if _tf.is_tensor(candidate):
                candidate = candidate.numpy()
        except Exception:
            pass
        preds_arr = np.array(candidate)
    else:
        try:
            import tensorflow as _tf
            if _tf.is_tensor(preds):
                preds_arr = preds.numpy()
            else:
                preds_arr = np.array(preds)
        except Exception:
            preds_arr = np.array(preds)

    probs = preds_arr[0] if preds_arr.ndim > 1 else preds_arr
    class_idx = int(np.argmax(probs))
    class_conf = float(probs[class_idx])

    # attention
    attention_score = _get_attention_score(grad_model, batch, class_idx)

    # YOLO detections
    impact_score = 0.0
    yolo_model = _load_local_yolo()
    detections_json = None
    if yolo_model is not None:
        try:
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

    if detections_json is None and YOLO_API_URL:
        detections_json = _call_external_yolo_api(image_path)

    if detections_json:
        impact_score = compute_impact_from_detections(detections_json)
    else:
        impact_score = 0.0

    chaos_score = _edge_density(arr_gray)
    base_severity = (attention_score * 0.5) + (impact_score * 0.3) + (chaos_score * 0.2)

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        out = analyze_disaster_image(sys.argv[1])
        print(json.dumps(out, indent=2))
    else:
        print("Call analyze_disaster_image('/path/to/image.jpg')")
