from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import os
import gdown
import tensorflow as tf

app = Flask(__name__)

# ==========================
# 1. Download model if not exists
# ==========================
model_path = "models/plant_disease_recog_model_pwp.keras"
model_file_id = "1_5G3Cz0WQtZeTxzsq5lrTUfEnNy78WeY"   # Google Drive model ID

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    os.makedirs("models", exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)

# ==========================
# 2. Load model + JSON
# ==========================
model = tf.keras.models.load_model(model_path)

# JSON is inside your repo (GitHub-safe)
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# ==========================
# 3. Flask Routes
# ==========================
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[str(prediction.argmax())]   # JSON keys are usually strings
    return prediction_label

@app.route('/upload/', methods=['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')
        prediction = model_predict(f'./{temp_name}_{image.filename}')
        return render_template('home.html', result=True, imagepath=f'/{temp_name}_{image.filename}', prediction=prediction)
    else:
        return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
