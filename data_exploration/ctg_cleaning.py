"""
ctg_cleaning.py
----------------
Preprocesses the CTG dataset for model training.
Performs:
- Missing value removal
- Column cleanup
- Log transforms
- Standardization
- Saves cleaned dataset as CSV

Usage:
    python data_exploration/ctg_cleaning.py
"""

# =============================
# Imports
# =============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =============================
# Load Dataset
# =============================
print("[INFO] Loading CTG dataset...")
df = pd.read_excel("CTG.xls", sheet_name="Raw Data")

# =============================
# Basic Cleaning
# =============================
print("[INFO] Cleaning dataset...")

# Remove empty rows and footer metadata
df = df.dropna(how="all")
df = df.drop(columns=[
    "FileName", "Date", "SegFile", "LBE", "A", "B", "C", "D", "E",
    "AD", "DE", "LD", "FS", "SUSP", "CLASS"
])
df = df.drop([2128, 2129], errors="ignore")  # remove non-data rows

# Adjust label values
df["NSP"] -= 1  # Convert {1,2,3} → {0,1,2}
df = df.drop_duplicates()

# =============================
# Feature Transformations
# =============================
print("[INFO] Applying transformations...")

# Log-transform skewed variables
logx = ["ASTV", "ALTV"]
for col in logx:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# Standardize all numeric features
scaler = StandardScaler()
feature_cols = [col for col in df.columns if col != "NSP"]
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# =============================
# Save Cleaned Dataset
# =============================
output_path = "data_exploration/CTG_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"[SUCCESS] Cleaned dataset saved to: {output_path}")
print(f"[INFO] Final shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")

