"""
inference.py
-------------
Loads a trained CTG model and makes predictions on new data.

Usage:
    python inference/inference.py --model models/xgb_model.pt --input new_ctg.csv
"""


# Imports

import argparse
import pandas as pd
import joblib
import numpy as np
import os


# Class Label Mapping

label_map = {
    0: "Normal",
    1: "Suspect",
    2: "Pathologic"
}


# Load Model + Scaler

def load_model(model_path, scaler_path="models/scaler.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    print(f"[INFO] Loading model from {model_path}")
    model = joblib.load(model_path)

    print(f"[INFO] Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    return model, scaler


# Load Input Data

def load_input(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded input data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# Predict Function

def predict(model, scaler, input_df):
    # Scale data (same as during training)
    scaled_data = scaler.transform(input_df)

    # Predict class
    preds = model.predict(scaled_data)
    pred_labels = [label_map[int(x)] for x in preds]

    # Create output DataFrame
    output = input_df.copy()
    output["Predicted_Class"] = preds
    output["Predicted_Label"] = pred_labels

    return output


# Main CLI

def main():
    parser = argparse.ArgumentParser(description="CTG Model Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.pt)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file with CTG data")
    parser.add_argument("--output", type=str, default="inference/predictions.csv", help="Output file path")

    args = parser.parse_args()

    model, scaler = load_model(args.model)
    input_df = load_input(args.input)

    preds = predict(model, scaler, input_df)
    preds.to_csv(args.output, index=False)

    print(f"[SUCCESS] Predictions saved to {args.output}")
    print(preds.head())


# Run Script
if __name__ == "__main__":
    main()
