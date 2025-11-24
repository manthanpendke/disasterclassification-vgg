# ---------- Process model output and present friendly info ----------
import json

def interpret_predictions(preds):
    """
    Handle several possible Keras predict outputs:
    - preds is a list/tuple: [class_probs, severity_scalar_or_array]
    - preds is a numpy array of shape (1, C): class probabilities
    - preds is a numpy array of shape (1,) or (): scalar -> treat as severity
    Returns: (class_idx, class_confidence, severity_percent)
    """
    preds_np = None
    # If model has multiple outputs (list/tuple), use first as class, second as severity if available
    if isinstance(preds, (list, tuple)):
        if len(preds) >= 2:
            class_out = np.asarray(preds[0])
            sev_out = np.asarray(preds[1])
            # class
            if class_out.ndim == 2:
                class_probs = class_out[0]
                class_idx = int(np.argmax(class_probs))
                class_conf = float(np.max(class_probs))
            else:
                # fallback
                class_idx = int(np.argmax(class_out))
                class_conf = 1.0
            # severity - reduce to scalar 0..1 if possible
            try:
                sev_val = float(np.squeeze(sev_out))
                # clamp 0..1
                sev_val = max(0.0, min(1.0, sev_val))
                severity_percent = sev_val * 100.0
            except Exception:
                severity_percent = class_conf * 100.0
            return class_idx, class_conf, severity_percent
        else:
            # single output wrapped in list
            preds_np = np.asarray(preds[0])
    else:
        preds_np = np.asarray(preds)

    # Now handle single-array cases
    # Case: classification probabilities -> shape (1, C)
    if preds_np.ndim == 2 and preds_np.shape[0] == 1 and preds_np.shape[1] > 1:
        class_probs = preds_np[0]
        class_idx = int(np.argmax(class_probs))
        class_conf = float(np.max(class_probs))
        severity_percent = class_conf * 100.0  # fallback estimate
        return class_idx, class_conf, severity_percent

    # Case: model returned a single severity scalar (shape (1,) or ())
    try:
        scalar = float(np.squeeze(preds_np))
        # if scalar seems like a probability 0..1, interpret as severity
        if 0.0 <= scalar <= 1.0:
            severity_percent = scalar * 100.0
            # No class info — return -1 as unknown
            return -1, 0.0, severity_percent
    except Exception:
        pass

    # Final fallback: try argmax on flattened array
    try:
        flat = preds_np.flatten()
        class_idx = int(np.argmax(flat))
        class_conf = float(np.max(flat))
        severity_percent = class_conf * 100.0
        return class_idx, class_conf, severity_percent
    except Exception:
        # give a default
        return -1, 0.0, 0.0

# call the interpreter
class_idx, class_conf, severity_percent = interpret_predictions(preds)

# load labels if present, otherwise use defaults
if Path("labels.txt").exists():
    with open("labels.txt", "r") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
else:
    # default labels (change if your dataset uses different names/order)
    labels = ["earthquake", "flood", "cyclone", "wildfire"]

if class_idx >= 0 and class_idx < len(labels):
    disaster_name = labels[class_idx]
else:
    disaster_name = "Unknown"

# responsibility mapping (tweak pairs to your needs)
authority_map = {
    "earthquake": "Local Municipality / Search & Rescue",
    "flood": "Disaster Management Authority / Coast Guard",
    "cyclone": "State Disaster Response Force / Meteorological Dept",
    "wildfire": "Forest Department / Local Fire Services"
}
responsible_authority = authority_map.get(disaster_name.lower(), "Local Municipality / Disaster Management Authority")

# severity label mapping
def severity_label(pct):
    if pct < 40.0:
        return "Low"
    elif pct < 80.0:
        return "Medium"
    else:
        return "High"

severity_label_str = severity_label(severity_percent)

# Present results in Streamlit
st.subheader("Inference result")
st.markdown(f"**Disaster:** {disaster_name}")
st.markdown(f"**Responsible authority:** {responsible_authority}")
st.markdown(f"**Severity:** {severity_percent:.1f}% ({severity_label_str})")
st.markdown(f"**Confidence (class):** {class_conf:.3f}")

# Provide machine-readable JSON output too
out = {
    "image_path": uploaded.name if hasattr(uploaded, "name") else "uploaded_image",
    "predicted_disaster": disaster_name,
    "class_confidence": round(class_conf, 4),
    "severity_percent": round(severity_percent, 3),
    "severity_label": severity_label_str,
    "responsible_authority": responsible_authority
}

st.code(json.dumps(out, indent=2))
