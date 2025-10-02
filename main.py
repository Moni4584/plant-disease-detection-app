import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# ==========================
# Constants
# ==========================
MODEL_URL = "https://drive.google.com/uc?id=1_liYB-Lv6HraFgDxS0WtSqVXTz1bwxBY"
MODEL_DIR = "models"
MODEL_FILENAME = "plant_disease_model_quant.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
IMG_SIZE = (160, 160)  # Make sure this matches your model input size

# ==========================
# Ensure model exists
# ==========================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024 * 1024:  # simple sanity check
    st.info("Downloading TFLite model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ==========================
# Load TFLite model safely
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
st.title("Plant Disease Detection App")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 160, 160, 3)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data)

    # Define your class labels (replace with your dataset labels)
    CLASS_LABELS = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry___Powdery_mildew',
        'Cherry___healthy',
        'Corn___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn___Common_rust',
        'Corn___Northern_Leaf_Blight',
        'Corn___healthy'
        # add all your classes here
    ]

    st.write(f"Predicted Class: **{CLASS_LABELS[pred_class]}**")
