# app.py
import streamlit as st
import traceback
import os
import json
from pathlib import Path
import tempfile

# Ensure repo root is importable and import your anna.py (uploaded file)
# anna.py must expose analyze_disaster_image(image_path)
try:
    import anna  # uses the uploaded anna.py you provided. See file used in this project.
except Exception as e:
    st.error("Import failed for `anna.py`. See details below.")
    st.stop()

st.set_page_config(page_title="Disaster Classifier (Anna)", layout="centered")

st.title("Disaster App — using anna.py backend")
st.caption("Uploads an image and runs `anna.analyze_disaster_image` prediction logic (no rewrite).")

# Button to delete cached model files (anna.py uses MODEL_LOCAL or LOCAL_FALLBACK)
MODEL_FILES = [
    "disaster_classifier_finetuned.keras",     # common MODEL_LOCAL name
    "/mnt/data/disaster_classifier_finetuned.keras",  # LOCAL_FALLBACK
    "disaster_classifier.keras",
    "/mnt/data/disaster_classifier.keras"
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

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# helper: severity label
def severity_label(percent: float) -> str:
    # percent is 0..100
    if percent < 40:
        return "Low"
    if percent < 80:
        return "Medium"
    return "High"

# run analysis and display
if uploaded is not None:
    # save to a temp file and call anna.analyze_disaster_image
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded.getbuffer())
        st.image(tmp_path, caption="Uploaded image", use_column_width=True)
        st.info("Running analysis (this may take a few seconds to load models)...")

        # call your analyze function
        try:
            result = anna.analyze_disaster_image(tmp_path)
        except Exception as e:
            st.error("Analysis failed. See traceback below.")
            st.code(traceback.format_exc())
            # cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # expected fields from your anna: predicted_class, class_confidence, estimated_severity, responsible_authority
            st.success("Analysis complete.")
            st.markdown("## Result")
            predicted = result.get("predicted_class", "unknown")
            conf = result.get("class_confidence", None)
            sev = result.get("estimated_severity", 0.0)
            auth = result.get("responsible_authority", "Local Authorities")
            # format
            sev_percent = round(float(sev)*100, 2)
            sev_level = severity_label(sev_percent)

            st.write(f"**Disaster:**  {predicted}")
            if conf is not None:
                st.write(f"**Class confidence:**  {conf}")
            st.write(f"**Estimated severity:**  {sev_percent}%")
            st.write(f"**Severity level:**  {sev_level}")
            st.write(f"**Responsible authority:**  {auth}")

            # show raw json for debugging
            with st.expander("Raw output JSON"):
                st.json(result)

            # optional: download result
            outfn = Path(tmp_path).stem + "_analysis.json"
            st.download_button("Download result JSON", json.dumps(result, indent=2), file_name=outfn, mime="application/json")

            # cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        st.error("Unexpected error while handling the uploaded file.")
        st.code(traceback.format_exc())
