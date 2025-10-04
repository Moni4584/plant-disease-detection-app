import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image

# ================================
# App Configuration
# ================================
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

MODEL_DIR = "models"
MODEL_FILE = "plant_disease_recog_model_pwp.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

GITHUB_MODEL_URL = "https://github.com/Moni4584/plant-disease-detection-app/releases/download/v.10/plant_disease_recog_model_pwp.keras"

os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# Download Model if not present
# ================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from GitHub..."):
        r = requests.get(GITHUB_MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)
    st.success("Model downloaded successfully!")

# ================================
# Load Model
# ================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# ================================
# Class Labels
# ================================
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ================================
# Preprocessing Function
# ================================
def preprocess_image(image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((160, 160))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================================
# Streamlit UI
# ================================
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the type of disease (or if it's healthy).")

uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"**Prediction:** {CLASS_NAMES[class_index]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        st.progress(float(confidence))

st.markdown("---")
st.markdown("Made with ❤️ using EfficientNetB4 and Streamlit")
