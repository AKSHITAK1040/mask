import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector.h5", compile=False)

model = load_model()

# Preprocessing (EXACT SAME AS WORKING CODE)
def preprocess_image(image):
    image = np.array(image)                # PIL → NumPy
    image = cv2.resize(image, (224, 224))  # resize
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("😷 Face Mask Detection")
st.write("Upload a face image to detect whether a mask is worn.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    pred = model.predict(processed, verbose=0)[0][0]

    # FINAL LABEL LOGIC (verified earlier)
    if pred > 0.5:
        label = "Mask"
        confidence = pred
        st.success(f"😷 MASK ({confidence*100:.1f}%)")
    else:
        label = "No Mask"
        confidence = 1 - pred
        st.error(f"❌ NO MASK ({confidence*100:.1f}%)")

    st.write(f"Raw prediction value: `{pred:.4f}`")
