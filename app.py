from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
import mahotas
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model, scaler, and label encoder once at startup
model_path = os.path.join(BASE_DIR, 'image_classification', 'output', 'rf_model.joblib')
scaler_path = os.path.join(BASE_DIR, 'image_classification', 'output', 'scaler.joblib')
label_encoder_path = os.path.join(BASE_DIR, 'image_classification', 'output', 'label_encoder.joblib')

clf = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le = joblib.load(label_encoder_path)

# Feature Extraction Function
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Color Histogram
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    for chan, color in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Hu Moments
    hu_moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    features.extend(hu_moments)

    # Haralick Features (Texture)
    textures = mahotas.features.haralick(gray)
    ht_mean = textures.mean(axis=0)
    features.extend(ht_mean)

    return features

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Image upload + prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        features = extract_features(image)

        # Pad or truncate to match expected input length
        if len(features) > 519:
            features = features[:519]
        elif len(features) < 519:
            features += [0] * (519 - len(features))

        features_scaled = scaler.transform([features])
        prediction = clf.predict(features_scaled)
        disease_name = le.inverse_transform(prediction)[0]

        image_url = f"/uploads/{file.filename}"

        return render_template('result.html', prediction=disease_name, image_url=image_url)

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render provides this env var
    app.run(host='0.0.0.0', port=port, debug=True)