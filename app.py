import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Fake & Deepfake Detection", layout="wide")

st.title("Fake News & Deepfake Detection Tool")
st.write("AI-powered detection for Fake News and Deepfake Images")

# ---------------------------
# Load Models (Cached)
# ---------------------------

@st.cache_resource
def load_fake_news_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_deepfake_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

fake_news_model = load_fake_news_model()
deepfake_model = load_deepfake_model()

# ---------------------------
# Helper Functions
# ---------------------------

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def detect_deepfake(image_path):
    image = preprocess_image(image_path)
    prediction = deepfake_model.predict(image)[0][0]
    confidence = round(float(prediction), 2)
    label = "FAKE" if confidence > 0.5 else "REAL"
    return label, confidence

# ---------------------------
# Fake News Detection
# ---------------------------

st.subheader("Fake News Detection")

news_text = st.text_area("Enter News Text")

if st.button("Analyze News"):
    if news_text.strip() == "":
        st.warning("Please enter text first.")
    else:
        with st.spinner("Analyzing text..."):
            result = fake_news_model(news_text)
            label = result[0]["label"]
            score = result[0]["score"]

            if label == "NEGATIVE":
                st.error(f"Likely FAKE (Confidence: {score:.2f})")
            else:
                st.success(f"Likely REAL (Confidence: {score:.2f})")

# ---------------------------
# Deepfake Image Detection
# ---------------------------

st.subheader("Deepfake Image Detection")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image = Image.open(uploaded_image).convert("RGB")
    image.save(temp_file.name)

    st.image(temp_file.name, width=300)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            label, confidence = detect_deepfake(temp_file.name)

            if label == "FAKE":
                st.error(f"Deepfake Detected (Confidence: {confidence:.2f})")
            else:
                st.success(f"Image Appears Real (Confidence: {1 - confidence:.2f})")

# ---------------------------
# Video Section Disabled (Cloud Safe)
# ---------------------------

st.subheader("Deepfake Video Detection")
st.info("Video detection is disabled in Streamlit Cloud for stability reasons.")

st.markdown("---")
st.markdown("Developed for AI-Based Fake & Deepfake Detection")
