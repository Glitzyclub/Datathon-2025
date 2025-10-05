"""
train_model.py
---------------
Main model training script for CTG classification.
Trains multiple models and saves the best ones with metrics.

Usage:
    python training/train_model.py
"""


# Imports

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Load and Prepare Data
print("[INFO] Loading cleaned dataset...")
df = pd.read_csv("data_exploration/CTG_cleaned.csv")

X = df.drop(columns=["NSP"])
y = df["NSP"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Ensure directory exists
os.makedirs("models", exist_ok=True)

# Define Evaluation Function
def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"\n=== {model_name.upper()} ===")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print(classification_report(y_test, preds))

    return bal_acc, f1


# Train Models

results = {}

# Logistic Regression
lr = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
lr.fit(X_train, y_train)
results["Logistic Regression"] = evaluate_model(lr, X_test, y_test, "Logistic Regression")
joblib.dump(lr, "models/lr_model.pt")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate_model(rf, X_test, y_test, "Random Forest")
joblib.dump(rf, "models/rf_model.pt")

# XGBoost
xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=40,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)
results["XGBoost"] = evaluate_model(xgb, X_test, y_test, "XGBoost")
joblib.dump(xgb, "models/xgb_model.pt")

# LightGBM
lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
results["LightGBM"] = evaluate_model(lgb, X_test, y_test, "LightGBM")
joblib.dump(lgb, "models/lgb_model.pt")

# CatBoost
ctb = CatBoostClassifier(verbose=0, random_state=42)
ctb.fit(X_train, y_train)
results["CatBoost"] = evaluate_model(ctb, X_test, y_test, "CatBoost")
joblib.dump(ctb, "models/ctb_model.pt")

# Neural Network (MLP) — with SMOTE balancing
print("\n[INFO] Applying SMOTE oversampling for Neural Net...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
nn.fit(X_res, y_res)
results["Neural Net"] = evaluate_model(nn, X_test, y_test, "Neural Net")
joblib.dump(nn, "models/nn_model.pt")


# Save Scaler
scaler = StandardScaler().fit(X)
joblib.dump(scaler, "models/scaler.pkl")


# Save Results Summary
summary = pd.DataFrame.from_dict(results, orient="index", columns=["Balanced Accuracy", "Macro F1"])
summary.to_csv("misc/model_comparison.csv")
print("\n=== Model Training Complete ===")
print(summary)
