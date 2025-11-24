# app.py
import streamlit as st
from PIL import Image
import tempfile, os, json
from pathlib import Path

st.set_page_config(page_title="Disaster Assessment", layout="centered")
st.title("Disaster Assessment (VGG + YOLO)")

# delete cached model button
if st.button("Delete cached model file (force re-download)"):
    try:
        if os.path.exists("disaster_classifier_finetuned.keras"):
            os.remove("disaster_classifier_finetuned.keras")
        st.success("Deleted cached model file. Rerun to download again.")
    except Exception as e:
        st.error(f"Could not delete cached model: {e}")

# upload
uploaded = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to get analysis (Disaster, Responsible authority, Severity%).")
    st.stop()

# save temp
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.image(Image.open(tmp_path), caption="Uploaded image", use_column_width=True)

# import anna (cleaned)
try:
    import anna
except Exception as e:
    st.error(f"Import failed: {e}")
    st.stop()

with st.spinner("Running analysis (this may take a few seconds to load models)..."):
    try:
        result = anna.analyze_disaster_image(tmp_path)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if result is None:
    st.error("Analysis returned no result.")
    st.stop()

# severity percent & label
sev = float(result.get("estimated_severity", 0.0))
# if value <=1 treat as 0..1
sev_pct = sev*100.0 if sev <= 1.0 else sev
def sev_label(p):
    if p < 40.0: return "Low"
    if p < 80.0: return "Medium"
    return "High"
label = sev_label(sev_pct)

# show results
st.subheader("Analysis Result")
st.markdown(f"**Disaster:** {result.get('predicted_class')}")
st.markdown(f"**Responsible authority:** {result.get('responsible_authority')}")
st.markdown(f"**Severity:** {sev_pct:.1f}% ({label})")
st.markdown(f"**Model confidence:** {result.get('class_confidence')}")

st.subheader("Raw JSON")
st.code(json.dumps(result, indent=2))
