import os
import sys
from typing import Dict, Tuple

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# Base directory so we can import local project modules reliably.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.sms_preprocessing_pipeline import DATASET_PATH, prepare_train_test_data


# Directory for trained SMS fraud models.
MODEL_DIR = os.path.join(BASE_DIR, "models")


def print_model_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> None:
    """Print consistent metrics for all models so comparison stays fair."""
    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def maybe_to_dense(matrix):
    """
    Convert sparse matrices to dense only when needed.
    Random Forest works more reliably with dense input on many installations.
    """
    if issparse(matrix):
        return matrix.toarray()
    return matrix


def build_logistic_regression() -> LogisticRegression:
    """
    Logistic Regression is a strong baseline for sparse TF-IDF text features.
    Regularization and balanced class weights help reduce overfitting.
    """
    return LogisticRegression(
        C=2.0,
        solver="lbfgs",
        l1_ratio=0.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )


def build_random_forest() -> RandomForestClassifier:
    """
    Random Forest gives a non-linear comparison model.
    Depth and leaf-size constraints are used to control overfitting.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=22,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )


def build_xgboost() -> XGBClassifier:
    """
    XGBoost is the main model because it can capture non-linear feature interactions
    while still handling high-dimensional feature spaces efficiently.
    Conservative hyperparameters plus early stopping help limit overfitting.
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=2.0,
        gamma=0.1,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=30,
    )


def save_model_bundle(filename: str, preprocessor, model, label_encoder, metrics: Dict[str, float]) -> None:
    """Save everything needed for later inference in one file."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "label_encoder": label_encoder,
            "metrics": metrics,
        },
        os.path.join(MODEL_DIR, filename),
    )


def evaluate_and_package(model_name: str, model, X_test_features, y_test: np.ndarray, class_names: np.ndarray) -> Dict[str, float]:
    """Run predictions and return a compact metrics dictionary for tracking."""
    y_pred = model.predict(X_test_features)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    }
    print_model_report(model_name, y_test, y_pred, class_names)
    return metrics


def train_all_models(dataset_path: str = DATASET_PATH) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    """
    Train all three models on the same preprocessed training split.
    This keeps the comparison fair across algorithms.
    """
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = prepare_train_test_data(dataset_path)

    # Fit the text/vector pipeline on training data only, then transform both splits.
    X_train_features = preprocessor.transform(X_train)
    X_test_features = preprocessor.transform(X_test)
    class_names = label_encoder.classes_

    trained_models: Dict[str, object] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}

    # 1. Logistic Regression baseline
    logistic_model = build_logistic_regression()
    logistic_model.fit(X_train_features, y_train)
    logistic_metrics = evaluate_and_package(
        "Logistic Regression",
        logistic_model,
        X_test_features,
        y_test,
        class_names,
    )
    save_model_bundle("sms_logistic_regression.pkl", preprocessor, logistic_model, label_encoder, logistic_metrics)
    trained_models["logistic_regression"] = logistic_model
    all_metrics["logistic_regression"] = logistic_metrics

    # 2. Random Forest comparison model
    rf_model = build_random_forest()
    X_train_rf = maybe_to_dense(X_train_features)
    X_test_rf = maybe_to_dense(X_test_features)
    rf_model.fit(X_train_rf, y_train)
    rf_metrics = evaluate_and_package(
        "Random Forest",
        rf_model,
        X_test_rf,
        y_test,
        class_names,
    )
    save_model_bundle("sms_random_forest.pkl", preprocessor, rf_model, label_encoder, rf_metrics)
    trained_models["random_forest"] = rf_model
    all_metrics["random_forest"] = rf_metrics

    # 3. XGBoost main model
    # We create a validation split from the training set for early stopping.
    X_train_main, X_valid, y_train_main, y_valid = train_test_split(
        X_train_features,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train,
    )

    xgb_model = build_xgboost()
    xgb_model.fit(
        X_train_main,
        y_train_main,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    xgb_metrics = evaluate_and_package(
        "XGBoost",
        xgb_model,
        X_test_features,
        y_test,
        class_names,
    )
    save_model_bundle("sms_xgboost.pkl", preprocessor, xgb_model, label_encoder, xgb_metrics)
    trained_models["xgboost"] = xgb_model
    all_metrics["xgboost"] = xgb_metrics

    print("\nModel Comparison Summary")
    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

    return trained_models, all_metrics


if __name__ == "__main__":
    train_all_models()
