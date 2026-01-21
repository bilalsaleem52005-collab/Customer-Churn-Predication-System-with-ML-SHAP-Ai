import os
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD DATA
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")

df = pd.read_csv(DATA_PATH)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Drop ID
df.drop(columns=["customerID"], inplace=True)

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# FEATURES & TARGET
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numeric features
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# XGBOOST MODEL
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,   # IMPORTANT for churn
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# Prediction 
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Although XGBoost showed slightly lower accuracy, it achieved high churn recall 
# by prioritizing at-risk customers. This trade-off is acceptable in churn prediction,
# where identifying potential churners is more important than overall accuracy.

import pickle
import os

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_churn_model.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ XGBoost model saved at:", MODEL_PATH)
