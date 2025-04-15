
<h1 align="center">🧠 Brain Tumor Detection System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-Flask-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ML-CNN%20%7C%20ANN%20%7C%20Classifiers-orange?style=for-the-badge"/>
</p>

---

## 📌 Overview

This project is an end-to-end Brain Tumor Detection System that integrates traditional Machine Learning models, Artificial Neural Networks (ANN), and Convolutional Neural Networks (CNN with Transfer Learning). It enables predictions both from **structured data** and **MRI images**, with a web interface built using **Flask**.

🎯 Goal: To provide a comprehensive, scalable, and production-ready tumor detection solution for medical diagnostic purposes.

---

## ⚙️ Technologies Used

- Python 3.8+
- Scikit-learn (ML models)
- Keras + TensorFlow (ANN, CNN)
- OpenCV, NumPy, Pandas
- Flask (Web Framework)
- HTML, CSS (UI)
- Matplotlib (Visualizations)

---

## 📂 Project Structure

```bash
Brain-Tumor-Detection-System/
├── .venv/                          # Virtual Environment
├── artifacts/                     # ML/ANN structured data (CSV) + model.pkl & preprocessor.pkl
├── braintumor3/                   # MRI Images for CNN model training
├── model_checkpoints_weights/    # CNN model weights, accuracy/loss checkpoints
├── logs/                          # Logging system (not fully maintained)
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Load and split dataset
│   │   ├── data_transformation.py # Preprocessing and feature engineering
│   │   └── model_trainer.py      # Trains multiple classifiers and saves best model
│   ├── pipelines/
│   │   └── predict_pipeline.py   # End-to-end ML prediction pipeline
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── home.html                 # Main Flask UI
│   └── index.html
│
├── app.py                        # Flask backend for user interaction
├── Final Project.ipynb           # Experimental Jupyter notebook
├── dl.py                         # ANN model for structured data
├── CNN.py                        # CNN & Transfer Learning for image classification
├── unsupervised/                 # Unsupervised learning approaches
├── requirements.txt              # Required packages
├── setup.py                      # Package configuration
└── README.md
```

---

## 🔍 ML & DL Models Used

### ✅ Supervised ML Models:
- Logistic Regression
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- AdaBoost

### 🔬 ANN:
- Built using Keras
- For structured CSV data
- Tuned with dropout and dense layers

### 🧠 CNN (CNN.py):
- Custom CNN with Transfer Learning
- Trained on `braintumor3/` images
- Model checkpoints saved for reproducibility

---

## 💻 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ankit1831/Brain-Tumor-Detection-System.git
cd Brain-Tumor-Detection-System
```

### 2️⃣ Setup the Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Web App

```bash
python app.py
```

Now, open your browser and go to `http://127.0.0.1:5000` 🎉

---

## 🖼 Sample UI

> Upload MRI scans or input symptoms to get real-time predictions from the best-trained model.

*(Add your own screenshots or GIFs here)*

---

## 📌 Future Improvements

- ✅ Add Grad-CAM visualization for CNN predictions
- 🐳 Dockerize the entire application
- 📊 Add Streamlit or Gradio interface
- ☁️ Deploy to AWS / Heroku / Render
- 📈 Model performance dashboard

---

## 🙋‍♂️ Author

**Ankit Singh**  
📬 [GitHub](https://github.com/ankit1831)

---

## ⭐ Show Your Support

If you liked this project, don’t forget to **⭐ star** the repo and share it with others!

```

