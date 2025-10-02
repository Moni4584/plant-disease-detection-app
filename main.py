from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import os
import gdown
import tensorflow as tf

# ==========================
# Flask app
# ==========================
app = Flask(__name__)

# ==========================
# 1. Ensure model is available
# ==========================
MODEL_DIR = "models"
MODEL_FILE = "plant_disease_recog_model_pwp.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_FILE_ID = "1_liYB-Lv6HraFgDxS0WtSqVXTz1bwxBY"  # Replace with your Google Drive file ID

# Create models directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# ==========================
# 2. Load Keras model
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# 3. Load JSON (local in repo)
# ==========================
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# ==========================
# 4. Flask Routes
# ==========================
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# ==========================
# 5. Helper functions
# ==========================
def extract_features(image_path):
    """Load image and convert to array for prediction."""
    # Load and ensure RGB
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)
    feature = feature / 255.0  # normalize to [0,1]
    return feature


def model_predict(image_path):
    """Predict plant disease from image."""
    img = extract_features(image_path)
    prediction = model.predict(img)
    prediction_label = plant_disease[str(prediction.argmax())]  # JSON keys must be strings
    return prediction_label

# ==========================
# 6. Upload Route
# ==========================
@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        # Ensure upload folder exists
        os.makedirs('uploadimages', exist_ok=True)

        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        saved_path = f'{temp_name}_{image.filename}'
        image.save(saved_path)

        prediction = model_predict(saved_path)
        return render_template('home.html', result=True, imagepath=f'/{saved_path}', prediction=prediction)
    else:
        return redirect('/')

# ==========================
# 7. Run Flask
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
