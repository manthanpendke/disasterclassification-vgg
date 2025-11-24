# anna.py  (clean inference module)
# Inference-only: VGG classifier (weights loaded from HuggingFace .keras) + YOLOv8 (local yolov8n.pt)
# Exposes: analyze_disaster_image(image_path) -> dict with keys:
#   image_path, predicted_class, class_confidence, estimated_severity, responsible_authority

import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf

# --- CONFIG ---
HF_MODEL_URL = "https://huggingface.co/manthanpendke/disaster-classifier-model/resolve/main/disaster_classifier_finetuned.keras"
LOCAL_FALLBACK = "/mnt/data/disaster_classifier_finetuned.keras"  # if running locally
MODEL_NAME_IN_INSTANCE = "disaster_classifier_finetuned.keras"
YOLO_LOCAL = "yolov8n.pt"  # keep yolov8n.pt in repo root

# Minimal size check (5MB)
_MIN_OK_SIZE = 5 * 1024 * 1024

# --- helper: download HF model (only if not present locally) ---
def _download_from_hf(url, dst_path):
    import requests, time
    session = requests.Session()
    resp = session.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(32768):
            if chunk:
                f.write(chunk)
    # quick sanity
    size = os.path.getsize(dst_path)
    if size < _MIN_OK_SIZE:
        raise RuntimeError(f"Downloaded model too small ({size} bytes).")
    # small sleep to be safe
    time.sleep(0.2)
    return dst_path

# --- Ensure model file available and return path ---
def _ensure_model_file():
    # prefer the explicit local fallback (useful in your dev env)
    if os.path.exists(LOCAL_FALLBACK):
        return LOCAL_FALLBACK
    if os.path.exists(MODEL_NAME_IN_INSTANCE) and os.path.getsize(MODEL_NAME_IN_INSTANCE) >= _MIN_OK_SIZE:
        return MODEL_NAME_IN_INSTANCE
    # otherwise download from HF
    _download_from_hf(HF_MODEL_URL, MODEL_NAME_IN_INSTANCE)
    return MODEL_NAME_IN_INSTANCE

# --- Lazy loaders (cached) ---
_classifier_model = None
_yolo_model = None
_idx_to_label = None
_IMG_SIZE = (224, 224)  # default fallback; will try to infer from model

def _load_classifier():
    global _classifier_model, _IMG_SIZE, _idx_to_label
    if _classifier_model is not None:
        return _classifier_model
    model_path = _ensure_model_file()
    model = tf.keras.models.load_model(model_path, compile=False)
    _classifier_model = model
    # try to infer input image size
    try:
        shape = model.inputs[0].shape
        # shape may be (None, H, W, C)
        h = int(shape[1]) if shape[1] is not None else None
        w = int(shape[2]) if shape[2] is not None else None
        if h and w:
            _IMG_SIZE = (w, h)
    except Exception:
        _IMG_SIZE = (224, 224)
    # attempt to extract labels if saved (optional)
    return _classifier_model

def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics library not installed. Add 'ultralytics' to requirements.") from e
    if not os.path.exists(YOLO_LOCAL):
        # if user didn't include yolov8n.pt in repo, let user know
        raise FileNotFoundError(f"{YOLO_LOCAL} not found in repo root. Please add yolov8n.pt.")
    _yolo_model = YOLO(YOLO_LOCAL)
    return _yolo_model

# --- Utility scoring functions (same logic as notebook) ---
def _get_attention_score(grad_model, img_array, class_idx):
    # img_array expected shape (1, H, W, C) and grad_model yields last conv + preds
    import tensorflow as tf
    with tf.GradientTape() as tape:
        last_conv, preds = grad_model(img_array)
        class_channel = preds[:, class_idx]
    grads = tape.gradient(class_channel, last_conv)
    if grads is None:
        return 0.0
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.squeeze(last_conv[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    if np.isnan(heatmap).any():
        return 0.0
    intensity = float(np.mean(heatmap))
    spread = float(np.sum(heatmap > (np.max(heatmap) * 0.5)) / heatmap.size) if np.max(heatmap) > 0 else 0.0
    return float((intensity * 0.5 + spread * 0.5) * 2.0)

def _get_yolo_impact_score(yolo_results, img_shape):
    # critical object IDs (COCO): 0 person, 2 car, 7 truck
    critical_object_ids = [0,2,7]
    total_critical_area = 0.0
    img_area = float(img_shape[0]) * float(img_shape[1])
    if not yolo_results:
        return 0.0
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    classes = yolo_results[0].boxes.cls.cpu().numpy()
    for i, box in enumerate(boxes):
        if int(classes[i]) in critical_object_ids:
            x1, y1, x2, y2 = box
            total_critical_area += float((x2 - x1) * (y2 - y1))
    density = total_critical_area / (img_area + 1e-9)
    score = min(1.0, density * 10.0)
    return float(score)

def _get_chaos_score(original_img_bgr):
    gray = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0)) / edges.size
    return float(min(1.0, edge_density * 4.0))

def _get_cyclone_eye_bonus(original_img_bgr):
    gray = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=10, maxRadius=100)
    return 0.3 if circles is not None else 0.0

# --- Main analysis function (exposed) ---
def analyze_disaster_image(image_path):
    """
    image_path: path to image file on disk (str)
    returns: dict with keys image_path, predicted_class, class_confidence,
             estimated_severity (0..1), responsible_authority
    """
    # ensure models loaded
    clf = _load_classifier()
    yolo = _load_yolo()

    # build classifier with attention grad model (reuse structure from notebook)
    # find last conv layer name (vgg16 block5_conv3 typical)
    try:
        last_conv = clf.get_layer("block5_conv3")
    except Exception:
        # try to locate a reasonable last conv
        for layer in reversed(clf.layers):
            if hasattr(layer, "kernel_size"):
                last_conv = layer
                break

    grad_model = tf.keras.Model(inputs=clf.inputs, outputs=[last_conv.output, clf.output])

    # read image
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        return None

    img_shape = original_bgr.shape  # (H, W, C)
    # prepare VGG input
    target_w, target_h = _IMG_SIZE
    img_vgg = cv2.resize(original_bgr, (target_w, target_h))
    img_array = img_vgg.astype(np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)

    # classifier prediction
    preds = clf.predict(batch, verbose=0)
    probs = preds[0]
    class_idx = int(np.argmax(probs))
    class_confidence = float(probs[class_idx])

    # compute scores
    attention_score = _get_attention_score(grad_model, batch, class_idx)
    # YOLO inference
    yolo_results = yolo(original_bgr, verbose=False)
    impact_score = _get_yolo_impact_score(yolo_results, img_shape)
    chaos_score = _get_chaos_score(original_bgr)

    base_severity = (attention_score * 0.5) + (impact_score * 0.3) + (chaos_score * 0.2)
    # cyclone bonus
    predicted_label = None
    # idx_to_label: try to infer class name mapping if not present
    global _idx_to_label
    if _idx_to_label is None:
        # best-effort: try to read labels.txt in repo
        labf = Path("labels.txt")
        if labf.exists():
            with open(labf, "r") as f:
                _idx_to_label = [l.strip() for l in f.readlines() if l.strip()]
        else:
            # fallback to default order used in your notebook — adjust if different
            _idx_to_label = ["earthquake", "flood", "cyclone", "wildfire"]

    if 0 <= class_idx < len(_idx_to_label):
        predicted_label = _idx_to_label[class_idx]
    else:
        predicted_label = f"class_{class_idx}"

    if predicted_label == "cyclone":
        base_severity += _get_cyclone_eye_bonus(original_bgr)

    final_severity = float(min(1.0, base_severity))

    # authority mapping same as notebook
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
        'class_confidence': round(class_confidence, 4),
        'estimated_severity': round(final_severity, 4),
        'responsible_authority': responsible_authority
    }
    return result

# convenience if run directly
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        out = analyze_disaster_image(sys.argv[1])
        print(json.dumps(out, indent=2))
    else:
        print("anna.py inference module. Call analyze_disaster_image(path).")
