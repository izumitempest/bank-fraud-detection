import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml_components import AlertStructuredFeaturesTransformer

# Configuration
DATASET_FILE = os.path.join(BASE_DIR, "data", "alerts_dataset_v2.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "alert_classifier_pipeline.pkl")
LABEL_NAMES = ["Legitimate", "Fake/Phishing", "Suspicious"]


def _get_text_frame(X):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=["text", "sender_id"])


def build_alert_pipeline():
    text_frame = FunctionTransformer(_get_text_frame, validate=False)

    feature_union = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "structured",
                Pipeline(
                    steps=[
                        ("extract", AlertStructuredFeaturesTransformer()),
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                ["text", "sender_id"],
            ),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        steps=[
            ("to_frame", text_frame),
            ("features", feature_union),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="saga",
                    multi_class="auto",
                    random_state=42,
                ),
            ),
        ]
    )


def load_alert_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Alert dataset must contain 'text' and 'label' columns.")

    if "sender_id" not in df.columns:
        df["sender_id"] = ""

    return df[["text", "sender_id", "label"]].copy()


def train_model():
    df = load_alert_dataset()
    X = df[["text", "sender_id"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_alert_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Alert Classifier Evaluation")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved alert pipeline to {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
