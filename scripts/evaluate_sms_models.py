import os
import re
import sys
from typing import Callable, Dict, List

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Base directory so local imports work when the script is run directly.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.sms_preprocessing_pipeline import (
    DATASET_PATH,
    FixedLabelEncoder,
    LABEL_TO_ID,
    build_preprocessing_pipeline,
    load_sms_dataset,
)
from scripts.train_sms_models import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    maybe_to_dense,
)

PLOT_DIR = os.path.join(BASE_DIR, "reports", "evaluation_charts")


def build_template_groups(text_series: pd.Series) -> pd.Series:
    """
    Normalize messages into coarse templates so nearly identical alerts stay in the same fold.
    This makes evaluation more realistic for template-heavy SMS datasets.
    """
    return (
        text_series.astype(str)
        .str.lower()
        .str.replace(r"\d+", "0", regex=True)
        .str.replace(r"[^a-z0\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def get_model_builders() -> Dict[str, Callable[[], object]]:
    """Return all models in one place so they use the same evaluation loop."""
    return {
        "Logistic Regression": build_logistic_regression,
        "Random Forest": build_random_forest,
        "XGBoost": build_xgboost,
    }


def fit_and_predict_fold(
    model_name: str,
    model_builder: Callable[[], object],
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    X_test_fold: pd.DataFrame,
) -> np.ndarray:
    """
    Train one model on one outer fold and return predictions for the held-out fold.
    The preprocessor is fit only on the training fold to avoid leakage.
    """
    preprocessor = build_preprocessing_pipeline()
    X_train_features = preprocessor.fit_transform(X_train_fold, y_train_fold)
    X_test_features = preprocessor.transform(X_test_fold)

    model = model_builder()

    if model_name == "Random Forest":
        model.fit(maybe_to_dense(X_train_features), y_train_fold)
        return model.predict(maybe_to_dense(X_test_features))

    if model_name == "XGBoost":
        # Keep a validation split inside the training fold for early stopping.
        X_train_main, X_valid, y_train_main, y_valid = train_test_split(
            X_train_features,
            y_train_fold,
            test_size=0.15,
            random_state=42,
            stratify=y_train_fold,
        )
        model.fit(
            X_train_main,
            y_train_main,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return model.predict(X_test_features)

    model.fit(X_train_features, y_train_fold)
    return model.predict(X_test_features)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> Dict[str, object]:
    """Compute the metrics required for model comparison."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))),
    }


def evaluate_models(dataset_path: str = DATASET_PATH, n_splits: int = 5) -> Dict[str, Dict[str, object]]:
    """
    Evaluate all models with grouped stratified cross-validation.
    Grouping by normalized alert template helps reduce train/test contamination.
    """
    df = load_sms_dataset(dataset_path)
    X = df[["sms_text", "sender_id"]].copy()
    label_encoder = FixedLabelEncoder(LABEL_TO_ID)
    y = label_encoder.transform(df["target_label"])
    class_names = label_encoder.classes_
    groups = build_template_groups(df["sms_text"])

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_results: Dict[str, Dict[str, object]] = {}

    for model_name, model_builder in get_model_builders().items():
        fold_predictions = np.empty_like(y)

        for train_idx, test_idx in cv.split(X, y, groups=groups):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X.iloc[test_idx]

            y_pred_fold = fit_and_predict_fold(
                model_name=model_name,
                model_builder=model_builder,
                X_train_fold=X_train_fold,
                y_train_fold=y_train_fold,
                X_test_fold=X_test_fold,
            )
            fold_predictions[test_idx] = y_pred_fold

        model_results[model_name] = calculate_metrics(y, fold_predictions, class_names)

    return model_results


def print_results(results: Dict[str, Dict[str, object]], class_names: np.ndarray) -> None:
    """Print a clean comparison table and the confusion matrix for each model."""
    print("SMS Fraud Model Evaluation")
    print("Evaluation method: 5-fold StratifiedGroupKFold on normalized alert templates")

    comparison_rows: List[Dict[str, object]] = []
    for model_name, metrics in results.items():
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1_score"], 4),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="F1 Score", ascending=False)
    print("\nComparison Table")
    print(comparison_df.to_string(index=False))

    for model_name, metrics in results.items():
        print(f"\n{model_name} Confusion Matrix")
        print(f"Labels: {list(class_names)}")
        print(metrics["confusion_matrix"])

    best_model_name = max(results.items(), key=lambda item: item[1]["f1_score"])[0]
    best_metrics = results[best_model_name]

    print("\nBest Performing Model")
    print(
        f"{best_model_name} "
        f"(Accuracy={best_metrics['accuracy']:.4f}, "
        f"Precision={best_metrics['precision']:.4f}, "
        f"Recall={best_metrics['recall']:.4f}, "
        f"F1={best_metrics['f1_score']:.4f})"
    )

    print("\nInterpretation")
    print(
        "These scores are more realistic than the earlier near-perfect holdout results because "
        "similar alert templates were grouped into the same fold before training. This reduces "
        "template leakage and gives a better estimate of how the models will behave on unseen SMS patterns."
    )


def save_metric_comparison_chart(results: Dict[str, Dict[str, object]]) -> str:
    """Save a grouped bar chart for model metric comparison."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(model_names))
    width = 0.18
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (metric_name, display_name, color) in enumerate(zip(metric_names, display_names, colors)):
        values = [results[model_name][metric_name] for model_name in model_names]
        bars = ax.bar(x + ((idx - 1.5) * width), values, width=width, label=display_name, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("SMS Fraud Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10)
    ax.set_ylim(0.0, 1.08)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(PLOT_DIR, "model_metrics_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_charts(results: Dict[str, Dict[str, object]], class_names: np.ndarray) -> List[str]:
    """Save one confusion matrix chart per model."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    saved_paths: List[str] = []

    for model_name, metrics in results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=class_names,
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{model_name} Confusion Matrix")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        output_path = os.path.join(PLOT_DIR, f"{safe_name}_confusion_matrix.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def save_f1_ranking_chart(results: Dict[str, Dict[str, object]]) -> str:
    """Save a simple ranking chart that highlights the best model."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    ranked = sorted(
        ((model_name, metrics["f1_score"]) for model_name, metrics in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    model_names = [item[0] for item in ranked]
    scores = [item[1] for item in ranked]
    colors = ["#2ca02c"] + ["#9ecae1"] * (len(model_names) - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, scores, color=colors)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.005,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Model Ranking by Weighted F1 Score")
    ax.set_ylabel("Weighted F1 Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(PLOT_DIR, "model_f1_ranking.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_all_evaluation_charts(results: Dict[str, Dict[str, object]], class_names: np.ndarray) -> None:
    """Generate and save all charts needed for the project report."""
    comparison_chart = save_metric_comparison_chart(results)
    confusion_charts = save_confusion_matrix_charts(results, class_names)
    ranking_chart = save_f1_ranking_chart(results)

    print("\nSaved Evaluation Charts")
    print(comparison_chart)
    for chart_path in confusion_charts:
        print(chart_path)
    print(ranking_chart)


if __name__ == "__main__":
    label_encoder = FixedLabelEncoder(LABEL_TO_ID)
    class_names = label_encoder.classes_
    evaluation_results = evaluate_models(DATASET_PATH, n_splits=5)
    print_results(evaluation_results, class_names)
    save_all_evaluation_charts(evaluation_results, class_names)


# ---------------------------------------------------------------------------
# Appended project evaluation chart block
# Assumes the following already exist before this block runs:
# models, X_test, y_test, X_train, y_train, vectorizer
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

import seaborn as sns
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

os.makedirs("charts", exist_ok=True)

CLASSES = ["Real", "Fake", "Suspicious"]
COLORS = ["#2563EB", "#DC2626", "#D97706"]
chart_count = 0


def _snake_case_model_name(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores
    raise AttributeError(f"{model.__class__.__name__} does not support predict_proba or decision_function.")


def _feature_names_for_importance(importances, fitted_vectorizer):
    feature_names = np.asarray(fitted_vectorizer.get_feature_names_out())
    if len(importances) == len(feature_names):
        return feature_names

    aligned_names = np.array([f"feature_{idx}" for idx in range(len(importances))], dtype=object)
    limit = min(len(feature_names), len(importances))
    aligned_names[:limit] = feature_names[:limit]
    return aligned_names


def _get_primary_vectorizer(preprocessor):
    feature_union = preprocessor.named_steps["feature_union"]
    return feature_union.named_transformers_["sms_tfidf"]


def _build_chart_runtime_objects():
    runtime_label_encoder = FixedLabelEncoder(LABEL_TO_ID)
    runtime_df = load_sms_dataset(DATASET_PATH)
    runtime_X = runtime_df[["sms_text", "sender_id"]].copy()
    runtime_y = runtime_label_encoder.transform(runtime_df["target_label"])

    runtime_X_train_df, runtime_X_test_df, runtime_y_train, runtime_y_test = train_test_split(
        runtime_X,
        runtime_y,
        test_size=0.2,
        random_state=42,
        stratify=runtime_y,
    )

    runtime_preprocessor = build_preprocessing_pipeline()
    runtime_X_train = runtime_preprocessor.fit_transform(runtime_X_train_df, runtime_y_train)
    runtime_X_test = runtime_preprocessor.transform(runtime_X_test_df)

    runtime_models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(),
    }

    runtime_models["Logistic Regression"].fit(runtime_X_train, runtime_y_train)
    runtime_models["Random Forest"].fit(maybe_to_dense(runtime_X_train), runtime_y_train)

    runtime_xgb_train, runtime_xgb_valid, runtime_y_train_main, runtime_y_valid = train_test_split(
        runtime_X_train,
        runtime_y_train,
        test_size=0.15,
        random_state=42,
        stratify=runtime_y_train,
    )
    runtime_models["XGBoost"].fit(
        runtime_xgb_train,
        runtime_y_train_main,
        eval_set=[(runtime_xgb_valid, runtime_y_valid)],
        verbose=False,
    )

    return (
        runtime_models,
        runtime_X_train,
        runtime_X_test,
        runtime_y_train,
        runtime_y_test,
        _get_primary_vectorizer(runtime_preprocessor),
    )


def _transform_for_model(model_name, features):
    if model_name == "Random Forest":
        return maybe_to_dense(features)
    return features


def _build_cv_estimator(model_name):
    if model_name == "Logistic Regression":
        return build_logistic_regression()
    if model_name == "Random Forest":
        return build_random_forest()
    estimator = build_xgboost()
    estimator.set_params(early_stopping_rounds=None, n_estimators=200)
    return estimator


if not all(name in globals() for name in ["models", "X_train", "X_test", "y_train", "y_test", "vectorizer"]):
    models, X_train, X_test, y_train, y_test, vectorizer = _build_chart_runtime_objects()


# CHART 1: ROC Curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

for model_name, model in models.items():
    model_name_snake = _snake_case_model_name(model_name)
    y_score = _predict_scores(model, _transform_for_model(model_name, X_test))

    fig, ax = plt.subplots(figsize=(7, 5))
    for class_index, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], y_score[:, class_index])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=COLORS[class_index], label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5)
    ax.set_title(f"ROC Curves — {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    filename = os.path.join("charts", f"roc_{model_name_snake}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(filename)
    chart_count += 1


# CHART 2: Precision-Recall Curves
for model_name, model in models.items():
    model_name_snake = _snake_case_model_name(model_name)
    y_score = _predict_scores(model, _transform_for_model(model_name, X_test))

    fig, ax = plt.subplots(figsize=(7, 5))
    for class_index, class_name in enumerate(CLASSES):
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin[:, class_index], y_score[:, class_index])
        pr_auc = auc(recall_vals, precision_vals)
        ax.plot(
            recall_vals,
            precision_vals,
            linewidth=2,
            color=COLORS[class_index],
            label=f"{class_name} (AUC = {pr_auc:.3f})",
        )

    ax.set_title(f"Precision-Recall Curves — {model_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()

    filename = os.path.join("charts", f"pr_{model_name_snake}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(filename)
    chart_count += 1


# CHART 3: Feature Importance
for model_name, model in models.items():
    if model_name not in {"XGBoost", "Random Forest"} or not hasattr(model, "feature_importances_"):
        continue

    model_name_snake = _snake_case_model_name(model_name)
    importances = model.feature_importances_
    feature_names = _feature_names_for_importance(importances, vectorizer)
    top_indices = np.argsort(importances)[-20:][::-1]
    top_features = feature_names[top_indices][::-1]
    top_values = importances[top_indices][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features, top_values, color="#2563EB", edgecolor="white")
    ax.set_title(f"Top 20 Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    filename = os.path.join("charts", f"feature_importance_{model_name_snake}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(filename)
    chart_count += 1


# CHART 4: Dataset Class Distribution
full_y = np.concatenate([y_train, y_test])
class_counts = pd.Series(full_y).value_counts().reindex([0, 1, 2], fill_value=0)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(CLASSES, class_counts.values, color=COLORS)
ax.bar_label(bars, padding=3)
ax.set_title("Dataset Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.tight_layout()

filename = os.path.join("charts", "class_distribution.png")
fig.savefig(filename, dpi=150)
plt.close(fig)
print(filename)
chart_count += 1


# CHART 5: Cross-Validation F1 Scores
cv_means = []
cv_stds = []
model_names = list(models.keys())

for model_name in model_names:
    estimator = _build_cv_estimator(model_name)
    scores = cross_val_score(
        estimator,
        _transform_for_model(model_name, X_train),
        y_train,
        cv=5,
        scoring="f1_macro",
    )
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, color="#2563EB", edgecolor="white")
for bar, mean_val, std_val in zip(bars, cv_means, cv_stds):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{mean_val:.3f} ± {std_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_title("5-Fold Cross-Validation F1 Scores")
ax.set_xlabel("Model")
ax.set_ylabel("Mean Macro F1 Score")
plt.tight_layout()

filename = os.path.join("charts", "cross_validation_f1.png")
fig.savefig(filename, dpi=150)
plt.close(fig)
print(filename)
chart_count += 1


# CHART 6: Learning Curves
train_sizes = np.linspace(0.1, 1.0, 8)

for model_name, model in models.items():
    model_name_snake = _snake_case_model_name(model_name)
    estimator = _build_cv_estimator(model_name)
    sizes, train_scores, valid_scores = learning_curve(
        estimator,
        _transform_for_model(model_name, X_train),
        y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_mean, color="#2563EB", linewidth=2, label="Training Score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, color="#2563EB", alpha=0.15)
    ax.plot(sizes, valid_mean, color="#DC2626", linewidth=2, label="CV Score")
    ax.fill_between(sizes, valid_mean - valid_std, valid_mean + valid_std, color="#DC2626", alpha=0.15)
    ax.set_title(f"Learning Curve — {model_name}")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Macro F1 Score")
    ax.legend(loc="lower right")
    plt.tight_layout()

    filename = os.path.join("charts", f"learning_curve_{model_name_snake}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(filename)
    chart_count += 1


# CHART 7: Combined Confusion Matrix (3-in-1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (model_name, model) in zip(axes, models.items()):
    y_pred = model.predict(_transform_for_model(model_name, X_test))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(model_name, fontweight="bold", fontsize=11)

fig.suptitle("Confusion Matrices — Model Comparison")
plt.tight_layout()

filename = os.path.join("charts", "all_confusion_matrices.png")
fig.savefig(filename, dpi=150)
plt.close(fig)
print(filename)
chart_count += 1


# CHART 8: Per-Class Metrics Heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, (model_name, model) in zip(axes, models.items()):
    y_pred = model.predict(_transform_for_model(model_name, X_test))
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    heatmap_df = pd.DataFrame(report).T.loc[CLASSES, ["precision", "recall", "f1-score"]]
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False)
    ax.set_title(model_name)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle("Per-Class Metrics Heatmap — All Models")
plt.tight_layout()

filename = os.path.join("charts", "classification_report_heatmap.png")
fig.savefig(filename, dpi=150)
plt.close(fig)
print(filename)
chart_count += 1


print("\n=== All evaluation charts saved to charts/ ===")
print(f"Total charts generated: {chart_count}")
