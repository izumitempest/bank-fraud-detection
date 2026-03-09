import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from imblearn.over_sampling import SMOTE

# Configuration
DATASET_PATH = "/home/izumi/Documents/CODE/Chichi/data/nibss_fraud_dataset.csv"
MODEL_PATH = "/home/izumi/Documents/CODE/Chichi/models/fraud_engine_model_v3.pkl"
ENCODER_PATH = "/home/izumi/Documents/CODE/Chichi/models/fraud_engine_encoders.pkl"
FEATURES_PATH = "/home/izumi/Documents/CODE/Chichi/models/fraud_engine_features.pkl"


def train_fraud_engine_v3():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    print("Loading NIBSS dataset...")
    df = pd.read_csv(DATASET_PATH)

    features = [
        "amount",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "tx_count_24h",
        "amount_sum_24h",
        "amount_mean_7d",
        "amount_std_7d",
        "tx_count_total",
        "amount_mean_total",
        "amount_std_total",
        "channel_diversity",
        "location_diversity",
        "amount_vs_mean_ratio",
        "online_channel_ratio",
        "channel",
        "merchant_category",
        "bank",
    ]

    X = df[features].copy()
    y = df["is_fraud"]

    print("Preprocessing & Encoding...")
    encoders = {}
    for col in ["channel", "merchant_category", "bank"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Training Level 2 Optimized Fraud Engine (XGBoost)...")
    # Using scale_pos_weight is another way to handle imbalance, but since we used SMOTE, we'll keep it simple
    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    clf.fit(X_train_res, y_train_res)

    print("\nAnalyzing Thresholds for Recall Optimization:")
    y_probs = clf.predict_proba(X_test)[:, 1]

    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05]
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        report = classification_report(y_test, y_pred_t, output_dict=True)
        recall = report["1"]["recall"]
        precision = report["1"]["precision"]
        print(f"Threshold: {t:.2f} | Recall: {recall:.2%} | Precision: {precision:.2%}")

    # We'll save the model and recommend a threshold of 0.2 in the API
    print(f"\nSaving optimized XGBoost model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(features, FEATURES_PATH)
    print("Optimization Level 2 Complete.")


if __name__ == "__main__":
    train_fraud_engine_v3()
