# app.py
import streamlit as st
from PIL import Image
import tempfile, os, json
import traceback

st.set_page_config(page_title="Disaster Assessment", layout="centered")
st.title("Disaster Assessment (VGG + YOLO)")

# show which external YOLO API is configured (if any)
YOLO_API_URL = os.environ.get("YOLO_API_URL", "")
if YOLO_API_URL:
    st.info("Using external YOLO API: configured")
else:
    st.info("Using local YOLO if available; otherwise external YOLO not configured.")

# allow deleting cached HF model file
if st.button("Delete cached model file (force re-download)"):
    try:
        if os.path.exists("disaster_classifier_finetuned.keras"):
            os.remove("disaster_classifier_finetuned.keras")
        if os.path.exists("/mnt/data/disaster_classifier_finetuned.keras"):
            os.remove("/mnt/data/disaster_classifier_finetuned.keras")
        st.success("Deleted cached model file. Rerun to download again.")
    except Exception as e:
        st.error(f"Could not delete cached model: {e}")

uploaded = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to get analysis (Disaster, Responsible authority, Severity%).")
    st.stop()

# Save uploaded image temporarily
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.image(Image.open(tmp_path), caption="Uploaded image", use_column_width=True)

# import anna (cleaned)
try:
    import anna
except Exception as e:
    st.error("Failed importing analyzer module 'anna'. See details below.")
    st.code(traceback.format_exc())
    st.stop()

# Run analysis
with st.spinner("Running analysis (this may take a few seconds to load models)..."):
    try:
        result = anna.analyze_disaster_image(tmp_path)
    except Exception as e:
        st.error("Analysis failed. See error trace below.")
        st.code(traceback.format_exc())
        st.stop()

if result is None:
    st.error("Analysis returned no result.")
    st.stop()

# Normalize severity to percentage (works for both 0..1 and 0..100 values)
sev_val = float(result.get("estimated_severity", 0.0))
sev_pct = sev_val * 100.0 if sev_val <= 1.0 else sev_val

def severity_label(p):
    if p < 40.0: return "Low"
    if p < 80.0: return "Medium"
    return "High"

sev_label = severity_label(sev_pct)

# Display outputs
st.subheader("Final Analysis")
st.write(f"**Disaster:** {result.get('predicted_class')}")
st.write(f"**Responsible Authority:** {result.get('responsible_authority')}")
st.write(f"**Severity:** {sev_pct:.1f}% ({sev_label})")
st.write(f"**Model Confidence:** {result.get('class_confidence')}")

st.subheader("Raw JSON Output")
st.code(json.dumps(result, indent=2))
