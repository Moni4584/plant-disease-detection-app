from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os
import requests

# ============================
# Flask App Configuration
# ============================
app = Flask(__name__)

MODEL_DIR = "models"
MODEL_FILE = "plant_disease_recog_model_pwp.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

GITHUB_MODEL_URL = "https://github.com/Moni4584/plant-disease-detection-app/releases/download/v.10/plant_disease_recog_model_pwp.keras"

# Ensure model folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Download Model if Not Present
# ============================
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from GitHub...")
    response = requests.get(GITHUB_MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")

# ============================
# Load Model
# ============================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ============================
# Labels
# ============================
label = [
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

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# ============================
# Routes
# ============================
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# ============================
# Helper Functions
# ============================
def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)
    feature = feature / 255.0  # normalize
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        os.makedirs("uploadimages", exist_ok=True)
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
        image.save(temp_name)
        prediction = model_predict(temp_name)
        return render_template('home.html', result=True, imagepath=f'/{temp_name}', prediction=prediction)
    else:
        return redirect('/')

# ============================
# Run the App
# ============================
if __name__ == "__main__":
    app.run(debug=True)

