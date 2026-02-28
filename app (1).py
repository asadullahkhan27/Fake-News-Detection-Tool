import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
from PIL import Image
from pytube import YouTube
import os
from PIL import Image
import tensorflow as tf
from transformers import pipeline
from tensorflow.keras.applications import Xception, EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---- Page Configuration ----
st.set_page_config(page_title="Fake & Deepfake Detection", layout="wide")

st.title("📰 Fake News & Deepfake Detection Tool")
st.write("🚀 Detect Fake News, Deepfake Images, and Videos using AI")

# Load Models
fake_news_detector = pipeline("text-classification", model="microsoft/deberta-v3-base")

@st.cache_resource
def load_fake_news_model():
    return pipeline("text-classification", model="microsoft/deberta-v3-base")

@st.cache_resource
def load_deepfake_models():
    base_model_image = Xception(weights="imagenet", include_top=False)
    base_model_image.trainable = False  
    x = GlobalAveragePooling2D()(base_model_image.output)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    deepfake_image_model = Model(inputs=base_model_image.input, outputs=x)

    base_model_video = EfficientNetB7(weights="imagenet", include_top=False)
    base_model_video.trainable = False
    x = GlobalAveragePooling2D()(base_model_video.output)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    deepfake_video_model = Model(inputs=base_model_video.input, outputs=x)

    return deepfake_image_model, deepfake_video_model

# Load models once in cache
fake_news_detector = load_fake_news_model()
deepfake_image_model, deepfake_video_model = load_deepfake_models()


# Load Deepfake Detection Models
base_model_image = Xception(weights="imagenet", include_top=False)
base_model_image.trainable = False  # Freeze base layers
x = GlobalAveragePooling2D()(base_model_image.output)
x = Dense(1024, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)  # Sigmoid for probability output
deepfake_image_model = Model(inputs=base_model_image.input, outputs=x)

base_model_video = EfficientNetB7(weights="imagenet", include_top=False)
base_model_video.trainable = False
x = GlobalAveragePooling2D()(base_model_video.output)
x = Dense(1024, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)
deepfake_video_model = Model(inputs=base_model_video.input, outputs=x)

# Function to Preprocess Image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100))  # Xception expects 299x299
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values
    return img

# Function to Detect Deepfake Image
def detect_deepfake_image(image_path):
    image = preprocess_image(image_path)
    prediction = deepfake_image_model.predict(image)[0][0]
    confidence = round(float(prediction), 2)
    label = "FAKE" if confidence > 0.5 else "REAL"
    return {"label": label, "score": confidence}

# ---- Fake News Detection Section ----
st.subheader("📝 Fake News Detection")
news_input = st.text_area("Enter News Text:", placeholder="Type here...")

# Manually verified facts database (you can expand this)
fact_check_db = {
    "elon musk was born in 1932": "FAKE",
    "earth revolves around the sun": "REAL",
    "the moon is made of cheese": "FAKE",
    "water boils at 100 degrees celsius": "REAL",
    "the great wall of china is visible from space": "FAKE",
    "human beings need oxygen to survive": "REAL",
    "vaccines cause autism": "FAKE",
    "the sun rises in the west": "FAKE",
    "chocolate is toxic to dogs": "REAL",
    "microsoft was founded by bill gates": "REAL",
    "dinosaurs and humans lived together": "FAKE",
    "the eiffel tower is in italy": "FAKE",
    "the speed of light is faster than sound": "REAL",
    "5g technology spreads covid-19": "FAKE",
    "honey never spoils": "REAL",
    "napoleon was extremely short": "FAKE",
    "goldfish have a three-second memory": "FAKE",
    "einstein failed math in school": "FAKE",
    "birds are descendants of dinosaurs": "REAL",
    "water is composed of hydrogen and oxygen": "REAL",
    "humans only use 10 percent of their brain": "FAKE",
    "the human body has 206 bones": "REAL",
    "the great pyramid of giza was built by aliens": "FAKE",
    "the internet was invented in 1983": "REAL",
    "earth is flat": "FAKE",
    "bananas grow on trees": "FAKE",
    "polar bears are left-handed": "FAKE",
    "the amazon rainforest produces 20 percent of the world's oxygen": "REAL",
    "dogs can see only black and white": "FAKE",
    "lightning never strikes the same place twice": "FAKE",
    "the shortest war lasted only 38 minutes": "REAL",
    "there is no gravity in space": "FAKE",
    "sharks do not get cancer": "FAKE",
    "the human heart beats about 100,000 times a day": "REAL",
    "albert einstein was a high school dropout": "FAKE",
    "diamonds are formed from coal": "FAKE",
    "the human tongue has different taste zones": "FAKE",
    "tomatoes are a fruit": "REAL",
    "a year on venus is shorter than a day": "REAL",
    "vikings wore horned helmets": "FAKE",
    "the moon has its own gravity": "REAL",
    "sugar causes hyperactivity in children": "FAKE",
    "human blood is blue inside the body": "FAKE",
    "gold is edible": "REAL",
    "ostriches bury their heads in the sand": "FAKE",
    "earth is the only planet with water": "FAKE",
    "black holes can evaporate": "REAL",
    "a penny dropped from the empire state building can kill a person": "FAKE",
    "octopuses have three hearts": "REAL",
    "mars is red because of iron oxide": "REAL",
    "eating carrots improves eyesight": "FAKE",
    "the human nose and ears keep growing with age": "REAL",
    "the leaning tower of pisa has always leaned": "REAL",
    "bats are blind": "FAKE",
    "you swallow eight spiders a year in your sleep": "FAKE",
    "the statue of liberty was a gift from france": "REAL",
    "light bulbs were invented by thomas edison": "REAL",
    "chameleons change color to match their surroundings": "FAKE",
    "dogs have unique nose prints": "REAL",
    "some frogs can survive being frozen": "REAL",
    "birds die if they eat rice": "FAKE",
    "a group of crows is called a murder": "REAL",
    "human dna is 60% similar to bananas": "REAL",
    "snakes can dislocate their jaws": "REAL",
    "the longest english word has 189,819 letters": "REAL",
    "there are more trees on earth than stars in the milky way": "REAL",
    "bananas are berries": "REAL",
    "peanuts are nuts": "FAKE",
    "avocados are poisonous to birds": "REAL",
    "a day on mercury is longer than its year": "REAL",
    "sharks existed before trees": "REAL",
    "the olympics were originally held in greece": "REAL",
    "human fingers have no muscles": "REAL",
    "cows have best friends": "REAL",
    "the inventor of the frisbee was turned into a frisbee after he died": "REAL",
    "watermelon is 92% water": "REAL",
    "new york was once called new amsterdam": "REAL",
    "the heart of a blue whale is the size of a small car": "REAL",
    "giraffes have the same number of neck bones as humans": "REAL",
    "venus is the hottest planet in the solar system": "REAL",
    "your hair and nails continue to grow after death": "FAKE",
    "the sun is a star": "REAL",
    "the human body glows in the dark but is invisible to the naked eye": "REAL",
    "barbie’s full name is barbara millicent roberts": "REAL",
    "ants can carry 50 times their own body weight": "REAL",
    "rabbits can’t vomit": "REAL",
    "the speed of sound is faster in water than in air": "REAL",
    "every planet in our solar system could fit between earth and the moon": "REAL",
    "a single lightning bolt is five times hotter than the sun’s surface": "REAL",
    "mosquitoes are the deadliest animals on earth": "REAL",
    "sea otters hold hands while sleeping": "REAL",
    "the empire state building can be seen from space": "FAKE",
    "your stomach gets a new lining every 3 to 4 days": "REAL",
    "hummingbirds can fly backward": "REAL",
    "a shrimp’s heart is in its head": "REAL",
    "the eiffel tower grows in the summer": "REAL",
    "neptune was the first planet discovered using math": "REAL"
}
def check_manual_facts(text):
    text_lower = text.lower().strip()
    return fact_check_db.get(text_lower, None)

if st.button("Check News"):
    # st.write("🔍 Processing...")
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.markdown("🔍 Processing...", unsafe_allow_html=True)

    # Check if the news is in the fact-check database
    manual_result = check_manual_facts(news_input)
    if manual_result:
        if manual_result == "FAKE":
            st.error(f"⚠️ Result: This news is **FAKE** (Verified by Database).")
        else:
            st.success(f"✅ Result: This news is **REAL** (Verified by Database).")
    else:
        # Use AI model if fact is not in the database
        prediction = fake_news_detector(news_input)
        label = prediction[0]['label'].lower()
        confidence = prediction[0]['score']

        if "fake" in label or confidence < 0.5:
            st.error(f"⚠️ Result: This news is **FAKE**. (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Result: This news is **REAL**. (Confidence: {confidence:.2f})")

st.subheader("📸 Deepfake Image Detection")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img = Image.open(uploaded_image).convert("RGB")
    img.save(temp_file.name, "JPEG")
    
    # 🛠️ FIX: Set image width (e.g., 300 pixels)
    st.image(temp_file.name, caption="🖼️ Uploaded Image", width=300)
    
    if st.button("Analyze Image"):
        st.write("🔍 Processing...")
        
        # Assuming detect_deepfake_image() is a function that returns a dictionary with "label" and "score"
        result = detect_deepfake_image(temp_file.name)
        
        if result["label"] == "REAL":
            st.success(f"✅ Result: This image is Real. (Confidence: {1 - result['score']:.2f})")
        else:
            st.error(f"⚠️ Result: This image is a Deepfake. (Confidence: {result['score']:.2f})")

 # ---- Deepfake Video Detection Section ----
st.subheader("🎥 Deepfake Video Detection")

# Upload video file
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# URL Input for Video (MP4 Direct Link or YouTube URL)
video_url = st.text_input("Enter Video URL (YouTube or MP4 Link)")

# Function to detect deepfake in video
def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_scores = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)
            result = detect_deepfake_image(frame_path)
            frame_scores.append(result["score"])
            os.remove(frame_path)
        
        frame_count += 1
    
    cap.release()
    
    if not frame_scores:
        return {"label": "UNKNOWN", "score": 0.0}
    
    avg_score = np.mean(frame_scores)
    confidence = round(float(avg_score), 2)
    final_label = "FAKE" if avg_score > 0.5 else "REAL"
    
    return {"label": final_label, "score": confidence}

# Download direct MP4 video
def download_video(url):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(temp_file.name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return temp_file.name
        else:
            return None
    except Exception as e:
        return None
        

# ✅ Display Video (YouTube Embed or Local)
video_path = None

if uploaded_video is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_file.name, "wb") as f:
        f.write(uploaded_video.read())
    video_path = temp_file.name  # Set video path for detection
    st.video(video_path)  # Show uploaded video

elif video_url:
    if "youtube.com" in video_url or "youtu.be" in video_url:
        st.markdown(
            f'<iframe width="560" height="315" src="{video_url.replace("watch?v=", "embed/")}" frameborder="0" allowfullscreen></iframe>',
            unsafe_allow_html=True,
        )
    else:
        video_path = download_video(video_url)  # Download MP4
        if video_path:
            st.video(video_path)  # Show downloaded MP4
        else:
            st.warning("⚠️ Invalid MP4 URL.")

# ✅ "Analyze Video" Button (Only for Local/MP4)
if uploaded_video or (video_url and not "youtube.com" in video_url):
    analyze_button = st.button("Analyze Video")

    if analyze_button and video_path:
        st.write("🔍 Processing... Please wait.")
        result = detect_deepfake_video(video_path)
        
        if result["label"] == "FAKE":
            st.error(f"⚠️ Deepfake Detected! This video appears to be FAKE. (Confidence: {result['score']:.2f})")
        elif result["label"] == "REAL":
            st.success(f"✅ This video appears to be REAL. (Confidence: {1 - result['score']:.2f})")
        else:
            st.warning("⚠️ Unable to analyze the video. Please try a different file.")

            
st.markdown("🔹 **Developed for Fake News & Deepfake Detection**") 
