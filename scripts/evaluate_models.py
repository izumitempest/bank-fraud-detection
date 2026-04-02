import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ALERT_DATASET = os.path.join(BASE_DIR, "data", "alerts_dataset_v2.csv")
ALERT_MODEL = os.path.join(BASE_DIR, "models", "alert_classifier_pipeline.pkl")
TRANSACTION_DATASET = os.path.join(BASE_DIR, "data", "nibss_fraud_dataset.csv")
TRANSACTION_MODEL = os.path.join(BASE_DIR, "models", "fraud_engine_pipeline_v3.pkl")

TRANSACTION_FEATURES = [
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


def print_metrics(name, y_true, y_pred, average="weighted"):
    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def evaluate_alert_model():
    if not os.path.exists(ALERT_DATASET) or not os.path.exists(ALERT_MODEL):
        print("Alert dataset or model is missing. Skipping alert evaluation.")
        return

    df = pd.read_csv(ALERT_DATASET)
    if "sender_id" not in df.columns:
        df["sender_id"] = ""

    X = df[["text", "sender_id"]]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load(ALERT_MODEL)
    y_pred = pipeline.predict(X_test)
    print_metrics("Alert Model Evaluation", y_test, y_pred, average="weighted")


def evaluate_transaction_model():
    if not os.path.exists(TRANSACTION_DATASET) or not os.path.exists(TRANSACTION_MODEL):
        print("Transaction dataset or model is missing. Skipping transaction evaluation.")
        return

    df = pd.read_csv(TRANSACTION_DATASET)
    X = df[TRANSACTION_FEATURES]
    y = df["is_fraud"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load(TRANSACTION_MODEL)
    y_pred = pipeline.predict(X_test)
    print_metrics("Transaction Model Evaluation", y_test, y_pred, average="binary")


if __name__ == "__main__":
    evaluate_alert_model()
    evaluate_transaction_model()
