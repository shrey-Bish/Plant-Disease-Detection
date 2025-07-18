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
    try:
        image = cv2.resize(image, (500, 500))  # same as training

        # Convert BGR → RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # RGB → HSV
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

        # Segmentation (same masks as training)
        lower_green = np.array([25, 0, 20])
        upper_green = np.array([100, 255, 255])
        healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)

        lower_brown = np.array([10, 0, 10])
        upper_brown = np.array([30, 255, 255])
        disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

        final_mask = healthy_mask + disease_mask
        segmented_img = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)

        # Hu Moments
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
        hu_moments = cv2.HuMoments(cv2.moments(gray)).flatten()

        # HSV Histogram (bins=8)
        hsv = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        features = np.hstack([hist, hu_moments])
        return features.tolist()
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


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

        if image is None:
            return render_template('result.html', prediction="Invalid image uploaded", image_url=None)

        features = extract_features(image)
        if features is None:
            return render_template('result.html', prediction="Feature extraction failed", image_url=None)

        # Pad/truncate feature vector
        expected_len = clf.n_features_in_
        if len(features) > expected_len:
            features = features[:expected_len]
        elif len(features) < expected_len:
            features += [0] * (expected_len - len(features))

        # Scale features
        features_scaled = scaler.transform([features])

        # Get prediction probabilities and class
        prediction_probs = clf.predict_proba(features_scaled)[0]
        prediction = clf.predict(features_scaled)[0]
        predicted_label = le.inverse_transform([prediction])[0]

        # Print debug info
        print("Prediction Probabilities:", prediction_probs)
        print("Class Mapping:", le.classes_)
        print("Predicted Class:", predicted_label)

        image_url = f"/uploads/{file.filename}"
        return render_template('result.html', prediction=predicted_label, image_url=image_url)

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5001)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))