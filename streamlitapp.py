import streamlit as st
import tempfile
from test import load_model, predict_video

st.title("🎭 Deepfake Detection System")

model = load_model("best_model.pt")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    st.write("Analyzing...")

    label, confidence = predict_video(model, tfile.name)

    if label is None:
        st.error("Could not read video.")
    else:
        if label == "FAKE":
            st.error(f"Prediction: FAKE")
        else:
            st.success(f"Prediction: REAL")

        st.write(f"Confidence: {confidence*100:.2f}%")
