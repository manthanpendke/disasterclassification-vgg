# app.py
import streamlit as st
import traceback
import os
import json
import tempfile
from pathlib import Path

# Import anna (deployment-safe module)
try:
    import anna
except Exception:
    st.set_page_config(page_title="Disaster Classifier (Anna)", layout="centered")
    st.title("Disaster App — import error")
    st.error("Import failed for anna.py. Check server logs for the full traceback.")
    st.code(traceback.format_exc())
    raise

st.set_page_config(page_title="Disaster Classifier (Anna)", layout="centered")
st.title("Disaster App")
st.caption("Uploads an image and runs anna.analyze_disaster_image prediction logic.")

# Model cache removal convenience
MODEL_FILES = [
    "disaster_classifier_finetuned.keras",
    "/mnt/data/disaster_classifier_finetuned.keras",
]

def delete_cached_models():
    deleted = []
    for p in MODEL_FILES:
        try:
            if os.path.exists(p):
                os.remove(p)
                deleted.append(p)
        except Exception:
            pass
    return deleted

if st.button("Delete cached model file (force re-download)"):
    deleted = delete_cached_models()
    if deleted:
        st.success(f"Deleted: {', '.join(deleted)}")
    else:
        st.info("No cached model files found at expected paths.")

st.markdown("### Upload an image (jpg/png/jpeg)")

# Provide non-empty label (accessibility warning suppression)
uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

def severity_label(percent: float) -> str:
    if percent < 40:
        return "Low"
    if percent < 80:
        return "Medium"
    return "High"

if uploaded is not None:
    # save uploaded to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        st.image(tmp_path, caption="Uploaded image", use_container_width=True)
        st.info("Running analysis (this may take a few seconds to load models)...")
        # call anna
        try:
            result = anna.analyze_disaster_image(tmp_path)
        except Exception:
            st.error("Analysis failed. See traceback below.")
            st.code(traceback.format_exc())
            raise

        # Present result
        st.success("Analysis complete.")
        st.markdown("## Result")
        predicted = result.get("predicted_class", "unknown")
        conf = result.get("class_confidence", None)
        sev = result.get("estimated_severity", 0.0)
        auth = result.get("responsible_authority", "Local Authorities")
        sev_percent = round(float(sev) * 100, 2) if float(sev) <= 1.0 else round(float(sev), 2)
        sev_level = severity_label(sev_percent)

        st.write(f"**Disaster:**  {predicted}")
        if conf is not None:
            st.write(f"**Class confidence:**  {conf}")
        st.write(f"**Estimated severity:**  {sev_percent}%")
        st.write(f"**Severity level:**  {sev_level}")
        st.write(f"**Responsible authority:**  {auth}")

        # Raw JSON
        with st.expander("Raw output JSON"):
            st.json(result)

        # Debug block if present
        if "debug" in result:
            with st.expander("Debug internals (from anna)"):
                st.json(result["debug"])

        # allow download
        outfn = Path(tmp_path).stem + "_analysis.json"
        st.download_button("Download result JSON", json.dumps(result, indent=2), file_name=outfn, mime="application/json")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
