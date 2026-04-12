import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.model_training import build_alert_pipeline
from scripts.dataset_generator import create_dataset

# Configuration
DATASET_FILE = os.path.join(BASE_DIR, "data", "alerts_dataset_v2.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "alert_classifier_pipeline.pkl")
LABEL_NAMES = ["Legitimate", "Fake/Phishing", "Suspicious"]


def load_dataset():
    if not os.path.exists(DATASET_FILE):
        data_dir = os.path.dirname(DATASET_FILE)
        os.makedirs(data_dir, exist_ok=True)
        print(f"Dataset not found: {DATASET_FILE}")
        print("Generating a synthetic alerts dataset for evaluation...")
        create_dataset()
        if not os.path.exists(DATASET_FILE):
            print("Failed to generate the dataset. Please check scripts/dataset_generator.py.")
            return None

    df = pd.read_csv(DATASET_FILE)
    if "text" not in df.columns or "label" not in df.columns:
        print("Alert dataset must contain 'text' and 'label' columns.")
        return None

    if "sender_id" not in df.columns:
        df["sender_id"] = ""

    return df[["text", "sender_id", "label"]].copy()


def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    pipeline = build_alert_pipeline()
    pipeline.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    return pipeline


def plot_metrics(metrics):
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [metrics[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=["#2E86AB", "#4CB5AE", "#F6C85F", "#A23E48"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Alert Model Performance Metrics (Weighted)")
    ax.set_ylabel("Score")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def evaluate():
    df = load_dataset()
    if df is None:
        return

    X = df[["text", "sender_id"]]
    y = df["label"]

    # Cross-validation for realistic performance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    base_pipeline = build_alert_pipeline()
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = clone(base_pipeline)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        fold_metrics.append((accuracy, precision, recall, f1))
        all_y_true.extend(list(y_test))
        all_y_pred.extend(list(y_pred))

    metrics_mean = {
        "Accuracy": sum(m[0] for m in fold_metrics) / len(fold_metrics),
        "Precision": sum(m[1] for m in fold_metrics) / len(fold_metrics),
        "Recall": sum(m[2] for m in fold_metrics) / len(fold_metrics),
        "F1-score": sum(m[3] for m in fold_metrics) / len(fold_metrics),
    }
    metrics_std = {
        "Accuracy": pd.Series([m[0] for m in fold_metrics]).std(),
        "Precision": pd.Series([m[1] for m in fold_metrics]).std(),
        "Recall": pd.Series([m[2] for m in fold_metrics]).std(),
        "F1-score": pd.Series([m[3] for m in fold_metrics]).std(),
    }

    print("Cross-Validation Metrics (mean ± std)")
    for key in metrics_mean:
        print(f"{key}: {metrics_mean[key]:.4f} ± {metrics_std[key]:.4f}")

    print("\nClassification Report (cross-val aggregated)")
    print(
        classification_report(
            all_y_true, all_y_pred, target_names=LABEL_NAMES, digits=4
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        all_y_true, all_y_pred, display_labels=LABEL_NAMES, cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    plot_metrics(metrics_mean)

    # Train on full data for final model + realistic test set
    pipeline = load_or_train_model(X, y)
    realistic_test = [
        ("Your GTBank acct was credited NGN5,000. Verify here: bit.ly/gtb-secure", 1),
        ("Zenith Alert: Debit NGN4,500 POS. If not you call support.", 2),
        ("UBA ALERT: CREDIT NGN25,000 Bal NGN150,000 Ref 123456", 0),
        ("Security notice: new device login. Ignore if this was you.", 2),
        ("Account restriction notice. Update BVN at gtb-secure.online", 1),
        ("Dear customer, beneficiary added. If not you, contact helpdesk.", 2),
    ]
    X_real = pd.DataFrame(
        [{"text": text, "sender_id": ""} for text, _ in realistic_test]
    )
    y_real = [label for _, label in realistic_test]
    y_real_pred = pipeline.predict(X_real)
    print("\nRealistic Test Set Report")
    print(
        classification_report(
            y_real, y_real_pred, target_names=LABEL_NAMES, digits=4
        )
    )

    classifier = pipeline.named_steps.get("classifier")
    model_name = classifier.__class__.__name__ if classifier else "Unknown"
    summary = (
        f"Model: {model_name}. The classifier is trained using a TF-IDF feature union "
        f"(word 1-2 grams + character 3-5 grams) plus structured text features, then "
        f"fit using stratified 5-fold cross-validation. This linear model is well-suited for SMS "
        f"fraud detection because it handles sparse high-dimensional text features "
        f"efficiently and is robust to noisy, short messages. Cross-validated weighted results: "
        f"Accuracy={metrics_mean['Accuracy']:.3f}±{metrics_std['Accuracy']:.3f}, "
        f"Precision={metrics_mean['Precision']:.3f}±{metrics_std['Precision']:.3f}, "
        f"Recall={metrics_mean['Recall']:.3f}±{metrics_std['Recall']:.3f}, "
        f"F1={metrics_mean['F1-score']:.3f}±{metrics_std['F1-score']:.3f}. "
        f"These results are more realistic than a single split, but synthetic datasets "
        f"still under-represent real-world linguistic diversity and evolving phishing tactics."
    )
    print("\nFinal Summary")
    print(summary)


if __name__ == "__main__":
    evaluate()
