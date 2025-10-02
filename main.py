import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# ==========================
# Constants
# ==========================
MODEL_URL = "https://drive.google.com/file/d/1YdaVH_sANHDPDLnXN8Xi76raEQWi3PzM/view?usp=drive_link"
MODEL_DIR = "models"
MODEL_FILENAME = "plant_disease_model_quant.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
IMG_SIZE = (160, 160)  # Make sure this matches your model input size

# ==========================
# Ensure model exists
# ==========================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024 * 1024:
    st.info("Downloading TFLite model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ==========================
# Load TFLite model
# ==========================
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load TFLite model: {e}")
    st.stop()

# ==========================
# Streamlit App UI
# ==========================
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ==========================
# Class labels
# ==========================
CLASS_LABELS = [
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
# Image prediction
# ==========================
if uploaded_file:
    image = Image.open(uploaded_file)

    # Convert grayscale to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 160, 160, 3)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get predicted class and confidence
    pred_index = np.argmax(output_data)
    confidence = float(output_data[pred_index])

    st.write(f"**Predicted Class:** {CLASS_LABELS[pred_index]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Display top 3 predictions
    top_indices = output_data.argsort()[-3:][::-1]
    st.write("### Top 3 Predictions:")
    for i in top_indices:
        st.write(f"{CLASS_LABELS[i]}: {output_data[i]*100:.2f}%")
