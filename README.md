# 🌿 Plant Disease Detection using Classical Machine Learning

> ⚠️ **Note:** For significantly better performance in image-based classification problems, consider using Deep Learning techniques like CNNs. This project demonstrates a classical Machine Learning pipeline as an educational and lightweight alternative.

---

## 📌 Project Overview

Plant disease detection plays a critical role in the agriculture industry. With the help of image processing and machine learning, we aim to identify whether a plant leaf is healthy or affected by diseases such as Apple Scab, Black Rot, or Cedar Apple Rust.

This project uses classical machine learning algorithms combined with image preprocessing and feature engineering to classify apple leaf images into healthy or diseased categories.

---

## 🧾 Dataset Information

- **Source**: [PlantVillage Dataset – Color Images](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)
- **Classes**: 
  - `Healthy` — Normal green apple leaves.
  - `Diseased` — Includes Apple Scab, Black Rot, and Cedar Apple Rust affected leaves.
- **Images Used**: 800 images each from `Healthy` and `Diseased` folders.

---

## 🖼️ Image Properties

| Property                   | Value         |
|---------------------------|---------------|
| File Type                 | JPG           |
| Dimensions                | 256 x 256     |
| Resolution                | 96 DPI        |
| Bit Depth                 | 24-bit Color  |

---

## ⚙️ Pipeline Steps

### 1. **Data Preprocessing**
- Load 800 images from each class.
- Convert images from RGB → BGR → HSV.
- Perform image segmentation to isolate the leaf from background.

### 2. **Feature Extraction**
- Extract **Color**, **Shape**, and **Texture** features using:
  - **Color**: Mean, Standard Deviation, Histogram
  - **Shape**: Hu Moments, Zernike Moments
  - **Texture**: Haralick Features, Local Binary Patterns (LBP)

### 3. **Feature Stacking and Encoding**
- Stack extracted features using NumPy.
- Encode labels (`Healthy`, `Diseased`) to numerical form.

### 4. **Train/Test Split**
- Data split in 80:20 ratio.

### 5. **Feature Scaling**
- Applied **Min-Max Scaling** to bring feature values between 0 and 1.

### 6. **Save Extracted Features**
- Stored features in **HDF5** format for efficient disk storage and later use.

### 7. **Modeling**
- Trained on 7 classical ML algorithms:
  - Logistic Regression
  - Linear Discriminant Analysis
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Naïve Bayes
  - Support Vector Machine
- **10-fold cross-validation** for model evaluation.

### 8. **Prediction**
- Best model (Random Forest) used for final predictions.
- Achieved **97.5% accuracy** on test set.

---

## 🧠 Result Summary

| Model               | Accuracy |
|--------------------|----------|
| Random Forest       | **97.5%** |
| Support Vector Machine | ~95% |
| KNN, Decision Trees | ~94–96% |
| Others              | ~90%+ |

---

## 📁 Project Structure

```bash
Plant-Disease-Detection/
├── utils/                       # Helper functions (label encoding, feature extraction)
├── Image Classification/
│   ├── train/                  # Training dataset with Healthy & Diseased folders
│   ├── Final_Notebook.ipynb    # Full end-to-end ML pipeline
├── Testing Notebook/
│   ├── testing.ipynb           # Detailed explanation of each function & test cases
├── models/                     # Saved models (optional)
├── features/                   # Saved HDF5 files of extracted features
└── README.md
