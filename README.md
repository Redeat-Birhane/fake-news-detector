# 🔍 Fake News Detector

> A machine learning web app that classifies news articles as **FAKE** or **REAL** using Logistic Regression and TF-IDF Vectorization — built with Python, Scikit-learn, and Flask.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## Overview

Fake News Detector addresses the growing challenge of misinformation by applying supervised machine learning to raw news text. Rather than relying on manual fact-checking, the system autonomously classifies news articles and headlines into **Real** or **Fake** categories and generates a confidence score — no journalist required at inference time.

---

## 🚀 Core Features

- **Scale-Agnostic Preprocessing** — Cleans raw news text by removing noise, stopwords, and applying Porter Stemming to normalize vocabulary across articles
- **TF-IDF Vectorization** — Converts cleaned text into numerical features by weighting terms based on their importance across the entire corpus
- **Logistic Regression Classifier** — Trains a lightweight, interpretable binary classifier that achieves high accuracy on the 44,000+ article dataset
- **Prescriptive Confidence Scoring** — Bypasses abstract probability outputs to return a human-readable prediction with a percentage confidence score
- **Flask Web Interface** — Serves predictions through a clean browser-based UI where users paste any headline or article and get an instant result

---

## 📊 Dataset Profile

The dataset consists of **44,898 news articles** across two CSV files monitoring real-world news content.

| File | Label | Count |
|------|-------|-------|
| `Fake.csv` | 0 — Fake | 23,481 |
| `True.csv` | 1 — Real | 21,417 |

📥 Download from Kaggle:
👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

> **💡 The Political Bias Pattern:** The model surfaces a counter-intuitive reality — fake news articles tend to be significantly longer than real ones, with more emotionally charged vocabulary. This is driven by sensationalist writing styles designed to maximize engagement over accuracy.

---

## 🛠️ Pipeline Architecture

```
Raw CSV Data (Fake.csv + True.csv)
        ↓
Label Assignment (0 = Fake, 1 = Real)
        ↓
Text Cleaning (utils.py — regex, stopwords, stemming)
        ↓
TF-IDF Vectorization (top 5,000 features)
        ↓
Train/Test Split (80/20)
        ↓
Logistic Regression Training
        ↓
Evaluation (Accuracy, F1, Confusion Matrix)
        ↓
Model Saved → models/model.pkl
        ↓
Flask App Loads Model → Serves Predictions via index.html
```

---

## 📁 Folder Structure

```
fake-news-detector/
├── data/
│   ├── Fake.csv                      # Raw fake news articles
│   └── True.csv                      # Raw real news articles
├── models/
│   ├── model.pkl                     # Trained logistic regression model
│   └── vectorizer.pkl                # Fitted TF-IDF vectorizer
├── notebooks/
│   └── fake_news_detection.ipynb     # Full ML pipeline with EDA
├── src/
│   └── utils.py                      # Shared text cleaning function
├── templates/
│   └── index.html                    # Web UI
├── app.py                            # Flask web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Quick Start & Execution

**Prerequisites**
```bash
pip install -r requirements.txt
```

**1. Download the Dataset**

Place `Fake.csv` and `True.csv` inside the `/data` folder.  
👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

**2. Train the Model**

Open and run all cells in:
```
notebooks/fake_news_detection.ipynb
```
This saves `model.pkl` and `vectorizer.pkl` to `/models`.

**3. Run the Web App**
```bash
python app.py
```
Open your browser at **http://localhost:5000**

---

## 🖥️ Sample Operational Output

When a news article is submitted, the engine skips abstract probability arrays and logs an explicit human-readable verdict:

```
=== FAKE NEWS DETECTION ENGINE: RUNNING INFERENCE ===

--- PREDICTION REPORT ---
Input Text  → "Government confirms 5G towers are spreading COVID-19 virus"
Prediction  → ❌ FAKE
Confidence  → 94.27%

--- PREDICTION REPORT ---
Input Text  → "NASA confirms Artemis II crew for first crewed lunar flyby"
Prediction  → ✅ REAL
Confidence  → 97.83%

======================================================
```

---

## ⚠️ Algorithmic Limitations & Next Steps

- **Domain Sensitivity** — The model is trained on political news data. Performance may degrade on sports, health, or entertainment misinformation due to vocabulary mismatch
- **Context Blindness** — TF-IDF treats each article as a bag of words and cannot capture sarcasm, tone, or contextual meaning the way transformer models can
- **Static Vocabulary** — The vectorizer freezes its 5,000-feature vocabulary at training time, meaning emerging slang or new terminology goes unrecognized at inference

**Future Work:**
- Migrating from Logistic Regression to transformer-based models (BERT, RoBERTa) for deeper semantic understanding
- Expanding the dataset to cover multi-domain misinformation beyond political news
- Adding a URL-based input so users can paste a news link directly instead of copying article text

---

## 🔗 Repository

```
https://github.com/Redeat-Birhane/fake-news-detector.git
```

---

## 🛠️ Tech Stack

- **Language** — Python 3.8+
- **ML Library** — Scikit-learn
- **NLP** — NLTK, TF-IDF
- **Web Framework** — Flask
- **Visualization** — Matplotlib, Seaborn
- **Notebook** — Jupyter
