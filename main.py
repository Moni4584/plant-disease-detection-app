import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import gdown  # for downloading from Google Drive

# ================================
# App Configuration
# ================================
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the type of disease (or if it's healthy).")

# ================================
# Model Setup
# ================================
MODEL_DIR = "models"
MODEL_FILE = "plant_disease_recog_model_pwp.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Google Drive file ID
GDRIVE_FILE_ID = "1_liYB-Lv6HraFgDxS0WtSqVXTz1bwxBY"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ================================
# Load Model
# ================================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

model = load_model()
st.success("✅ Model loaded successfully!")

# ================================
# Class Labels
# ================================
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ================================
# Image Preprocessing
# ================================
def preprocess_image(image):
    """Convert image to RGB, resize to 160x160, normalize, add batch dimension"""
    image = image.convert("RGB")
    image = image.resize((160, 160))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================================
# Streamlit UI
# ================================
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"🌱 Prediction: {CLASS_NAMES[class_index]}")
        st.info(f"Confidence: {confidence*100:.2f}%")
        st.progress(float(confidence))

st.markdown("---")
st.markdown("💡 **Made with ❤️ using EfficientNetB4 and Streamlit**")
