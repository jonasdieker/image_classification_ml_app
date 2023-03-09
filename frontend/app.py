import json

import requests
import streamlit as st

from backend.inference import class_names

GOOGLE_CREDENTIALS = ""
PROJECT = ""
REGION = ""


def predict_json(filename):
    """Uses backend to post image and receives a prediction."""

    files = {"image": filename.read()}
    # headers = {"content_type": filename.type}
    r = requests.post(url="http://backend:8080/predict", files=files)
    pred_class, pred_conf = json.loads(r.text).values()
    return pred_class, pred_conf


def make_prediction(image):
    """Takes image from session_state, loads it and predicts its class.""" 
    pred_class, pred_conf = predict_json(image)
    return pred_class, pred_conf


def main():
    """main function for running streamlit frontend."""
    
    st.title("First ML App")
    st.header("Classify what's in your photos!")

    if st.checkbox("Show supported classes"):
        st.write(f"These are the classes the model can identify\n: {class_names}")

    # upload file
    uploaded_file = st.file_uploader(label="Upload image", type=["png", "jpeg", "jpg"])

    if not uploaded_file:
        st.warning("Please upload an image.")
        st.stop()
    else:
        st.session_state["uploaded_image"] = uploaded_file
        st.image(st.session_state["uploaded_image"], use_column_width=True)
        pred_button = st.button("Predict", key="pred_button")

    if st.session_state.pred_button:
        st.session_state.pred_class, st.session_state.pred_conf = make_prediction(uploaded_file)
        st.write(f"Prediction: {st.session_state.pred_class},\
                 Confidence: {st.session_state.pred_conf:.3f}")


if __name__ == "__main__":
    main()
