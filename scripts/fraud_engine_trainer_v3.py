import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
DATASET_PATH = os.path.join(BASE_DIR, "data", "nibss_fraud_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_engine_pipeline_v3.pkl")

NUMERIC_FEATURES = [
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
]

CATEGORICAL_FEATURES = ["channel", "merchant_category", "bank"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"


def build_fraud_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.3,
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def load_transaction_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Transaction dataset is missing required columns: {missing}")

    return df[FEATURES + [TARGET]].copy()


def train_fraud_engine_v3():
    df = load_transaction_dataset()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_fraud_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Transaction Fraud Engine Evaluation")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved fraud pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    train_fraud_engine_v3()
