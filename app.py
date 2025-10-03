from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os
import requests

# ==========================
# CONFIG
# ==========================
MODEL_DIR = "models"
MODEL_FILE = "plant_disease_recog_model_pwp.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# GitHub release download link
MODEL_URL = "https://github.com/Moni4584/plant/releases/download/v1.0.0/plant_disease_recog_model_pwp.keras"

# Make sure models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# DOWNLOAD MODEL IF NOT PRESENT
# ==========================
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from GitHub Releases...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded successfully!")

# ==========================
# LOAD MODEL
# ==========================
app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# LABELS
# ==========================
label = [
 'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
 'Background_without_leaves','Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy',
 'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
 'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy',
 'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
 'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# ==========================
# ROUTES
# ==========================
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        os.makedirs("uploadimages", exist_ok=True)
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        save_path = f'{temp_name}_{image.filename}'
        image.save(save_path)
        prediction = model_predict(save_path)
        return render_template('home.html', result=True, imagepath=f'/{save_path}', prediction=prediction)
    else:
        return redirect('/')

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
