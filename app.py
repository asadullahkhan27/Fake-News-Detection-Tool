import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
import os
from PIL import Image
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake & Deepfake Detection", layout="wide")
st.title("Fake News & Deepfake Detection Tool")
st.write("Detect Fake News, Deepfake Images, and Videos using AI")

# -------------------------------
# Load Models (Cached)
# -------------------------------

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

fake_news_detector = load_fake_news_model()
deepfake_model = load_deepfake_model()

# -------------------------------
# Helper Functions
# -------------------------------

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def detect_deepfake_image(image_path):
    image = preprocess_image(image_path)
    prediction = deepfake_model.predict(image)[0][0]
    confidence = round(float(prediction), 2)
    label = "FAKE" if confidence > 0.5 else "REAL"
    return {"label": label, "score": confidence}

# -------------------------------
# Fake News Detection
# -------------------------------

st.subheader("Fake News Detection")
news_input = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            prediction = fake_news_detector(news_input)
            label = prediction[0]['label']
            confidence = prediction[0]['score']

            if label == "NEGATIVE":
                st.error(f"Result: Likely FAKE (Confidence: {confidence:.2f})")
            else:
                st.success(f"Result: Likely REAL (Confidence: {confidence:.2f})")

# -------------------------------
# Deepfake Image Detection
# -------------------------------

st.subheader("Deepfake Image Detection")
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image = Image.open(uploaded_image).convert("RGB")
    image.save(temp_file.name)

    st.image(temp_file.name, width=300)

    if st.button("Analyze Image"):
        with st.spinner("Processing Image..."):
            result = detect_deepfake_image(temp_file.name)

            if result["label"] == "FAKE":
                st.error(f"Deepfake Detected (Confidence: {result['score']:.2f})")
            else:
                st.success(f"Image Appears Real (Confidence: {1 - result['score']:.2f})")

# -------------------------------
# Deepfake Video Detection
# -------------------------------

st.subheader("Deepfake Video Detection")
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_scores = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)
            result = detect_deepfake_image(frame_path)
            frame_scores.append(result["score"])
            os.remove(frame_path)

        frame_count += 1

    cap.release()

    if len(frame_scores) == 0:
        return {"label": "UNKNOWN", "score": 0.0}

    avg_score = np.mean(frame_scores)
    confidence = round(float(avg_score), 2)
    label = "FAKE" if avg_score > 0.5 else "REAL"

    return {"label": label, "score": confidence}

if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    st.video(video_path)

    if st.button("Analyze Video"):
        with st.spinner("Processing Video..."):
            result = detect_deepfake_video(video_path)

            if result["label"] == "FAKE":
                st.error(f"Deepfake Detected (Confidence: {result['score']:.2f})")
            else:
                st.success(f"Video Appears Real (Confidence: {1 - result['score']:.2f})")

st.markdown("---")
st.markdown("Developed for AI-Based Fake & Deepfake Detection")
