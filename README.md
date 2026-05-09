# ❤️ ECG-Based Deep Learning System for Heart Disease Classification

## 📌 Overview

Cardiovascular diseases (CVDs) are one of the leading causes of death worldwide. ECG (Electrocardiogram) analysis is a crucial diagnostic tool, but traditional manual interpretation is time-consuming, subjective, and prone to noise and artifacts in scanned reports.

This project presents a **hybrid deep learning + machine learning framework** that automatically classifies ECG images into multiple heart disease categories with high accuracy.

---

## 🚀 Key Features

* 🔄 **Image-to-Signal Transformation** (ECG Image → 1D Signal)
* 🧠 **Vision Transformer (ViT)** for deep feature extraction
* ⚙️ **Stacked Ensemble Learning** (SVC + Logistic Regression)
* 📉 **PCA for Dimensionality Reduction** (3060 → ~400 features)
* 📊 High performance with **96.77% accuracy**
* 💻 Simple UI for ECG image upload and prediction

---


---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:** NumPy, OpenCV, Scikit-learn, TensorFlow/PyTorch
* **Techniques:** PCA, Ensemble Learning, Vision Transformer
* **Tools:** Jupyter Notebook, VS Code

---

## 📊 Dataset

* **Source:** Mendeley 12-Lead ECG Image Dataset
* **Total Records:** 929 ECG images
* **Classes:**

  * Normal
  * Myocardial Infarction (MI)
  * Abnormal Heartbeat
  * Prior MI

---

## 🔬 Methodology

### Step 1: Image Preprocessing

* Grayscale conversion
* Noise removal (Gaussian + Morphological filtering)
* Otsu thresholding
* Lead segmentation

### Step 2: Image → Signal Conversion

* Contour-based waveform extraction
* Convert ECG to **1D signal (3060 features)**

### Step 3: Feature Optimization

* Standardization
* PCA → reduce to ~400 features

### Step 4: Model Training

* Vision Transformer (ViT)
* SVC + Logistic Regression

### Step 5: Stacked Ensemble

* Combine predictions using meta-learner

---

## 📈 Results


✅ **Best Model:** Stacked Ensemble (ViT + SVC + Logistic Regression)

---

## 🖥️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Tanishq0505/ECG-based-deep-learning-system-to-classify-heart-diseases.git
cd ECG-based-deep-learning-system-to-classify-heart-diseases
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Project

```bash
streamlit run app.py
```

---

## 📸 Demo Flow

1. Upload ECG Image
2. Preprocessing & Noise Removal
3. Image → Signal Conversion
4. Feature Reduction (PCA)
5. Model Prediction
6. Output: Predicted Class

---

## 🎯 Applications

* 🏥 Clinical decision support systems
* 📱 Telemedicine & remote diagnosis
* 🔬 Medical research & diagnostics
* ⚡ Automated ECG interpretation

---

## ⚠️ Limitations

* Limited dataset size
* Computational complexity
* Not yet deployed on cloud

---

## 🔮 Future Work

* Larger and diverse datasets
* Cloud deployment (AWS/GCP)
* Real-time ECG analysis
* Integration with clinical data

---

## 👨‍💻 Contributors

* **Tanishq Anand** – System design & preprocessing
* **Riya Dublish** – Dataset Collection Evaluation & testing
* **Sumit Sahni** – Ensemble learning implementation
* **Siddharth Pathak** – Vit Model development

---

## 📜 Research Paper

📄 *A Hybrid Vision Transformer Stacked Ensemble Framework for Automated Multi-Class ECG Image Classification*

---

## ⭐ Acknowledgment

Special thanks to our guide **Dr. Rahul Gupta** for guidance and support.

---

## 📬 Contact


🔗 GitHub: https://github.com/Tanishq0505

---

