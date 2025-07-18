# 🌿 Plant Disease Detection using Classical Machine Learning

> ⚠️ **Note:** For better performance in image-based classification problems, Deep Learning (e.g., CNNs) is recommended. This project demonstrates a classical Machine Learning pipeline as a lightweight, educational alternative.

---

## 🌐 Live Demo

👉 [Click here to try it live on Render](https://plant-disease-detection.onrender.com)


---

## 📌 Project Overview

This project helps identify plant leaf diseases using image processing + classical machine learning techniques. It classifies uploaded apple leaf images into **Healthy** or **Diseased** categories (e.g., Apple Scab, Black Rot, Cedar Apple Rust).

✅ Upload a leaf image on the web interface
✅ View real-time prediction result
✅ Uses classical ML models — lightweight and fast

---

## 🧾 Dataset Information

* **Source**: [PlantVillage – Color Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)
* **Classes**:

  * `Healthy` — Normal apple leaves.
  * `Diseased` — Includes Apple Scab, Black Rot, Cedar Apple Rust.
* **Images Used**: 800 per class (balanced)

---

## 🖼️ Image Properties

| Property     | Value        |
| ------------ | ------------ |
| Type         | JPG          |
| Size         | 256x256 px   |
| Resolution   | 96 DPI       |
| Color Format | 24-bit Color |

---

## ⚙️ ML Pipeline Summary

### 1. **Data Preprocessing**

* Load and resize images
* Convert RGB → HSV → Segment leaf
* Background removal for clean feature extraction

### 2. **Feature Extraction**

Extracted using custom utilities:

* **Color Features**: HSV Histogram, mean, std
* **Shape Features**: Hu Moments
* **Texture Features**: Local Binary Patterns (LBP)

### 3. **Feature Engineering**

* Combined features into a single vector
* Label encoding
* Min-Max Scaling applied
* Stored as `.h5` for quick loading

### 4. **Modeling**

Tested 7 ML algorithms:

* Logistic Regression
* Linear Discriminant Analysis
* KNN
* Decision Tree
* Random Forest ✅ (Best)
* Naïve Bayes
* Support Vector Machine

✅ **Random Forest achieved 97.5% accuracy**
✅ Cross-validation (10-fold) used for reliability

---

## 🧪 Result Summary

| Model                  | Accuracy  |
| ---------------------- | --------- |
| ✅ Random Forest        | **97.5%** |
| Support Vector Machine | \~95%     |
| KNN, Decision Trees    | 94–96%    |
| Others                 | 90%+      |

---

## 💻 Web App Features

* 🌱 Simple Flask UI
* 📤 Upload your own leaf image
* 📸 Preview rendered image before prediction
* ✅ Model prediction displayed on `result.html` page
* 🗃️ Automatically saves image uploads to `uploads/`
* ⚡ Fast inference (no reloading model/scaler each time)

---

## 🚀 Deployment

Deployed via **Render**.
Just push your code and it auto-deploys Flask with model loading.

### 🛠️ `requirements.txt` includes:

```txt
Flask
scikit-learn
numpy
joblib
opencv-python
matplotlib>=3.7.1
```

Add Render build commands:

```
Build Command: pip install -r requirements.txt
Start Command: python app.py
```

---

## 📁 Project Structure

```
Plant-Disease-Detection/
├── app.py                     # Flask backend
├── uploads/                   # Stores uploaded leaf images
├── image_classification/
│   ├── dataset/               # Train/test images
│   ├── output/                # Saved models (.joblib)
│   └── Final_Notebook.ipynb  # Full ML pipeline
├── utils/                     # Feature extractors & helpers
├── templates/
│   ├── index.html             # Upload UI
│   └── result.html            # Displays predicted label + image
├── static/
│   └── styles.css (optional) # Styling and uploaded image rendering
├── requirements.txt
└── README.md
```

---

## 🙌 Acknowledgments

* [PlantVillage Dataset by Mohanty et al.](https://github.com/spMohanty/PlantVillage-Dataset)
* Used OpenCV, Scikit-learn, and Flask

---

