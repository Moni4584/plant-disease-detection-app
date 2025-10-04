import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from PIL import Image

# =====================================
# App Configuration
# =====================================
st.set_page_config(page_title="🌿 Plant Disease Detection", page_icon="🍃", layout="centered")

MODEL_DIR = "models"
MODEL_FILE = "plant_disease_recog_model_pwp.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# 🔹 GitHub Release Model URL (your v.10 release)
GITHUB_MODEL_URL = (
    "https://github.com/Moni4584/plant-disease-detection-app/releases/download/v.10/"
    "plant_disease_recog_model_pwp.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================
# Download Model if Not Present
# =====================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Downloading model from GitHub..."):
        response = requests.get(GITHUB_MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    st.success("✅ Model downloaded successfully!")

# =====================================
# Load Model
# =====================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("⚠️ Error loading model. Please verify the file path or model format.")
    st.exception(e)
    st.stop()

# =====================================
# Class Labels
# =====================================
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
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

# =====================================
# Preprocessing Function
# =====================================
def preprocess_image(image):
    """Convert any image to RGB, resize to (160,160,3), normalize."""
    # Ensure RGB (handles grayscale too)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to match model input
    image = image.resize((160, 160))

    # Convert to numpy array
    img_array = np.array(image) / 255.0

    # Add batch dimension → (1,160,160,3)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# =====================================
# Streamlit UI
# =====================================
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect whether it’s healthy or infected with a disease.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        try:
            with st.spinner("Analyzing image..."):
                processed = preprocess_image(image)
                prediction = model.predict(processed)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

            # Display results
            st.success(f"🌱 **Prediction:** {CLASS_NAMES[class_index]}")
            st.info(f"🔹 **Confidence:** {confidence * 100:.2f}%")
            st.progress(float(confidence))

        except Exception as e:
            st.error("⚠️ An error occurred during prediction. Please check your image.")
            st.exception(e)

# =====================================
# Footer
# =====================================
st.markdown("---")
st.markdown(
    "💡 **Made with ❤️ using EfficientNetB4 and Streamlit**  \n"
    "📁 Model release: [v.10 on GitHub](https://github.com/Moni4584/plant-disease-detection-app/releases/tag/v.10)"
)
