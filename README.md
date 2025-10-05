# 🩺 CTG Classification — Datathon 2025

## 📘 Overview
This project was developed for **Datathon 2025**, focusing on the detection and classification of **Cardiotocography (CTG)** scans into three fetal health states:

- **Normal (0)** — Healthy fetal state  
- **Suspect (1)** — Potential warning signals  
- **Pathologic (2)** — Fetal distress  

The goal is to design an automated pipeline that can process raw CTG data, clean and standardize it, train multiple ML models, and evaluate their performance in identifying fetal health conditions.

## 🧭 Repository Structure

> Datathon-2025/ \
│ \
├── README.md \
├── report.docx ← Academic report (methodology + findings)
│
├── data_exploration/
│ ├── ctg_exploration.ipynb ← Exploratory data analysis & visualization
│ └── ctg_cleaning.py ← Data preprocessing & feature scaling
│
├── training/
│ └── train_model.py ← Main model training script
│
├── models/
│ ├── nn_model.pt ← Saved neural net weights
│ ├── rf_model.pt ← Saved RandomForest weights
│ └── scaler.pkl ← Fitted StandardScaler for inference
│
├── inference/
│ └── inference.py ← Predicts fetal state on new CTG samples
│
└── misc/
├── confusion_matrix.png
├── feature_importance.png
└── model_comparison.csv


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

> git clone https://github.com/Glitzyclub/Datathon-2025.git
>
> cd Datathon-2025

### 2️⃣ Install Dependencies

> pip install -r requirements.txt

## 🚀 How to Run
### 🧹 Step 1: Data Exploration
Open the notebook:

> data_exploration/ctg_exploration.ipynb

This file performs:

- Data visualization and feature analysis
- Correlation heatmaps
- Missing value checks
- Skew correction and scaling

### 🧠 Step 2: Train Models

> python training/train_model.py

This script:

- Cleans and scales the CTG dataset
- Trains multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Neural Net)
- Evaluates each using:
  - Balanced Accuracy
  - Macro F1 Score
- Saves the best-performing models under /models/.

### 🔎 Step 3: Run Inference
Once trained, test the model on new unseen data:

> python inference/inference.py

This script:

- Loads the saved model and scaler
- Processes new CTG samples
- Outputs predicted fetal states (0 = Normal, 1 = Suspect, 2 = Pathologic)

## 📊 Model Performance Summary
|  Model	|  Balanced Accuracy	| Macro F1 | Key Observation |
| --------|---------------------|----------|-----------------|
| Logistic Regression |	0.87	| 0.86	| Good interpretability |
| Random Forest |	0.97 |	0.98	| Strong overall performance |
| XGBoost |	0.96	|  0.97	| Slightly better minority class handling |
| Neural Net (MLP) |	0.74 |	0.78 |	Underperforms due to data imbalance |

## 💡 Insights
Most false negatives occur when Suspect cases are predicted as Normal, aligning with real-world clinical ambiguity.

Tree-based models (RF, XGB) outperform the neural net on limited data.

Logistic Regression provides interpretability, aiding in explainable AI reporting.



