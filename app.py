# app.py (Streamlit wrapper)
import streamlit as st
from PIL import Image
import tempfile
import os
from pathlib import Path
import json

st.set_page_config(page_title="Disaster Assessment", layout="centered")

st.title("Disaster Assessment (VGG + YOLO)")

# Provide a button to clear cached model file in instance
if st.button("Delete cached model file (force re-download)"):
    try:
        if os.path.exists("disaster_classifier_finetuned.keras"):
            os.remove("disaster_classifier_finetuned.keras")
        st.success("Deleted cached model file. Rerun to download again.")
    except Exception as e:
        st.error(f"Could not delete cached model: {e}")

uploaded = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to get analysis (Disaster, Responsible authority, Severity%).")
    st.stop()

# Save uploaded image to temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.image(Image.open(tmp_path), caption="Uploaded image", use_column_width=True)

# Import the cleaned anna module
try:
    import anna
except Exception as e:
    st.error(f"Import failed: {e}")
    st.stop()

# Run analysis
with st.spinner("Running analysis (this may take a few seconds to load models)..."):
    try:
        result = anna.analyze_disaster_image(tmp_path)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if result is None:
    st.error("Analysis returned no result.")
    st.stop()

# Convert severity to percent and label
sev_pct = float(result.get("estimated_severity", 0.0)) * 100.0 if result.get("estimated_severity", 0.0) <= 1.0 else float(result.get("estimated_severity", 0.0))
def sev_label(p):
    if p < 40.0: return "Low"
    if p < 80.0: return "Medium"
    return "High"

label = sev_label(sev_pct)

# Display outputs
st.subheader("Analysis Result")
st.markdown(f"**Disaster:** {result.get('predicted_class')}")
st.markdown(f"**Responsible authority:** {result.get('responsible_authority')}")
st.markdown(f"**Severity:** {sev_pct:.1f}% ({label})")
st.markdown(f"**Model confidence:** {result.get('class_confidence')}")

# JSON output
st.subheader("Raw JSON")
st.code(json.dumps(result, indent=2))
