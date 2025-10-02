# ==========================
# main.py
# Plant Disease Detection (Streamlit + TFLite + Google Drive)
# ==========================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ==========================
# 1. Ensure model is available (Google Drive)
# ==========================
MODEL_DIR = "models"
MODEL_FILE = "plant_disease_model_quant.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# 🔑 Replace with your own TFLite Google Drive file ID
MODEL_FILE_ID = "1_liYB-Lv6HraFgDxS0WtSqVXTz1bwxBY"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading TFLite model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False
    )

# ==========================
# 2. Load TFLite model
# ==========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (height, width)

# ==========================
# 3. Class labels (39 classes)
# ==========================
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ==========================
# 4. Helper functions
# ==========================
def preprocess_image(image: Image.Image):
    image = image.resize(input_shape)
    image = np.array(image, dtype=np.float32)
    if image.shape[-1] == 4:  # RGBA → RGB
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)
    class_index = np.argmax(prediction)
    confidence = prediction[class_index] * 100
    return CLASS_NAMES[class_index], confidence, prediction

# ==========================
# 5. Streamlit UI
# ==========================
st.set_page_config(page_title="🌱 Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection (TFLite Model)")
st.write("Upload a plant leaf image and the app will predict the disease using a TFLite model from Google Drive.")

uploaded_file = st.file_uploader("📂 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Prediction
    label, confidence, prediction = predict(image)

    st.success(f"✅ Predicted Disease: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Show top 5 predictions
    top5_idx = np.argsort(prediction)[-5:][::-1]
    st.subheader("🔎 Top 5 Predictions")
    for i in top5_idx:
        st.progress(float(prediction[i]))
        st.write(f"**{CLASS_NAMES[i]}** → {prediction[i]*100:.2f}%")
