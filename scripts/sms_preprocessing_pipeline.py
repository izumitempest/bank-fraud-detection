import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


# Update this path to point to your SMS alert dataset file.
DATASET_PATH = os.path.join("data", "alerts_dataset_v2.csv")

# Fixed class names for the fraud alert task.
EXPECTED_LABELS = ["Real", "Fake", "Suspicious"]
LABEL_TO_ID = {"Real": 0, "Fake": 1, "Suspicious": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

# Common fallback names so the script can work across slightly different CSV schemas.
TEXT_COLUMN_CANDIDATES = ["sms", "sms_text", "message", "message_text", "text", "body"]
SENDER_COLUMN_CANDIDATES = ["sender_id", "sender", "sender_name", "source", "origin"]
LABEL_COLUMN_CANDIDATES = ["label", "class", "target", "category"]

# Lightweight stopword list using scikit-learn only.
STOPWORDS = set(ENGLISH_STOP_WORDS)

# Simple regex patterns for SMS cleaning and structure features.
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|ng|xyz|link|site)\b)", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"(\bngn\b|\bnaira\b|[#$£€₦]|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)", re.IGNORECASE)
ACCOUNT_PATTERN = re.compile(r"\*{2,}\d+|\bacct\b|\baccount\b", re.IGNORECASE)
SUSPICIOUS_TERMS = (
    "verify",
    "click",
    "urgent",
    "suspended",
    "blocked",
    "update",
    "otp",
    "pin",
    "password",
    "bvn",
    "nin",
    "link",
)
BANK_TERMS = (
    "gtbank",
    "gtb",
    "zenith",
    "uba",
    "access",
    "opay",
    "kuda",
    "fidelity",
    "fcmb",
    "sterling",
    "wema",
    "firstbank",
)


def find_first_matching_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    """Return the first column whose lowercase name matches any candidate."""
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_sender(sender: object) -> str:
    """Standardize sender IDs so missing values and case variations are handled consistently."""
    if pd.isna(sender):
        return "unknown_sender"

    sender_text = str(sender).strip().lower()
    if not sender_text:
        return "unknown_sender"

    sender_text = re.sub(r"\s+", " ", sender_text)
    return sender_text


def clean_sms_text(text: object) -> str:
    """
    Clean SMS text by:
    1. Lowercasing
    2. Removing special characters
    3. Removing stopwords
    4. Returning a whitespace-normalized string
    """
    if pd.isna(text):
        text = ""

    text = str(text).lower()
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    tokens = tokenize_sms_text(text)
    return " ".join(tokens)


def tokenize_sms_text(text: str) -> List[str]:
    """Split text into tokens and remove stopwords and one-character noise tokens."""
    if not text:
        return []

    tokens = text.split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def normalize_target_label(value: object) -> str:
    """
    Map raw dataset labels into the required classes:
    Real, Fake, Suspicious.
    Supports both string labels and numeric encodings.
    """
    if pd.isna(value):
        raise ValueError("Target label contains missing values. Please fix labels before training.")

    raw = str(value).strip().lower()
    label_map: Dict[str, str] = {
        "0": "Real",
        "1": "Fake",
        "2": "Suspicious",
        "real": "Real",
        "legitimate": "Real",
        "genuine": "Real",
        "normal": "Real",
        "fake": "Fake",
        "phishing": "Fake",
        "fraud": "Fake",
        "fraudulent": "Fake",
        "spam": "Fake",
        "suspicious": "Suspicious",
        "suspect": "Suspicious",
        "warning": "Suspicious",
    }

    if raw not in label_map:
        raise ValueError(
            f"Unsupported label '{value}'. Expected labels that can map to {EXPECTED_LABELS}."
        )

    return label_map[raw]


def load_sms_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the CSV file and standardize the core columns used in the pipeline."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    text_column = find_first_matching_column(df.columns, TEXT_COLUMN_CANDIDATES)
    label_column = find_first_matching_column(df.columns, LABEL_COLUMN_CANDIDATES)
    sender_column = find_first_matching_column(df.columns, SENDER_COLUMN_CANDIDATES)

    if text_column is None:
        raise ValueError(
            f"Could not find an SMS text column. Checked: {TEXT_COLUMN_CANDIDATES}"
        )

    if label_column is None:
        raise ValueError(
            f"Could not find a target label column. Checked: {LABEL_COLUMN_CANDIDATES}"
        )

    standardized = pd.DataFrame(
        {
            "sms_text": df[text_column],
            "sender_id": df[sender_column] if sender_column else "",
            "target_label": df[label_column],
        }
    )

    # Handle missing values before any splitting or feature creation.
    standardized["sms_text"] = standardized["sms_text"].fillna("")
    standardized["sender_id"] = standardized["sender_id"].fillna("unknown_sender")
    standardized["target_label"] = standardized["target_label"].apply(normalize_target_label)
    return standardized


class SMSFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create cleaned text, tokenized text strings, and structured SMS features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=["sms_text", "sender_id"])

        df["sms_text"] = df["sms_text"].fillna("").astype(str)
        df["sender_id"] = df["sender_id"].apply(normalize_sender)

        # Clean and tokenize the SMS text. TF-IDF will be fit later on train data only.
        df["clean_text"] = df["sms_text"].apply(clean_sms_text)
        df["tokenized_text"] = df["clean_text"].apply(lambda text: " ".join(tokenize_sms_text(text)))
        df["sender_clean"] = df["sender_id"].apply(clean_sms_text)

        # Add sender and message structure features that often help fraud detection.
        df["message_length"] = df["sms_text"].str.len()
        df["word_count"] = df["sms_text"].str.split().str.len().fillna(0)
        df["digit_count"] = df["sms_text"].apply(lambda value: sum(char.isdigit() for char in value))
        df["uppercase_ratio"] = df["sms_text"].apply(self._uppercase_ratio)
        df["punctuation_count"] = df["sms_text"].apply(self._punctuation_count)
        df["has_url"] = df["sms_text"].apply(lambda value: int(bool(URL_PATTERN.search(value))))
        df["has_amount"] = df["sms_text"].apply(lambda value: int(bool(AMOUNT_PATTERN.search(value))))
        df["has_account_reference"] = df["sms_text"].apply(lambda value: int(bool(ACCOUNT_PATTERN.search(value))))
        df["suspicious_term_count"] = df["sms_text"].apply(self._suspicious_term_count)
        df["mentions_bank"] = df["sms_text"].apply(self._mentions_bank)
        df["sender_is_numeric"] = df["sender_id"].apply(lambda value: int(value.replace(" ", "").isdigit()))
        df["sender_length"] = df["sender_id"].str.len()

        return df

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return 0.0
        uppercase = sum(char.isupper() for char in letters)
        return uppercase / len(letters)

    @staticmethod
    def _punctuation_count(text: str) -> int:
        return sum(not char.isalnum() and not char.isspace() for char in text)

    @staticmethod
    def _suspicious_term_count(text: str) -> int:
        lowered = text.lower()
        return sum(term in lowered for term in SUSPICIOUS_TERMS)

    @staticmethod
    def _mentions_bank(text: str) -> int:
        lowered = text.lower()
        return int(any(term in lowered for term in BANK_TERMS))


def build_preprocessing_pipeline() -> Pipeline:
    """Build the full preprocessing pipeline for SMS fraud classification."""
    structured_feature_columns = [
        "message_length",
        "word_count",
        "digit_count",
        "uppercase_ratio",
        "punctuation_count",
        "has_url",
        "has_amount",
        "has_account_reference",
        "suspicious_term_count",
        "mentions_bank",
        "sender_is_numeric",
        "sender_length",
    ]

    feature_union = ColumnTransformer(
        transformers=[
            (
                "sms_tfidf",
                TfidfVectorizer(
                    tokenizer=str.split,
                    preprocessor=None,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
                "tokenized_text",
            ),
            (
                "sender_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(2, 4),
                    min_df=1,
                ),
                "sender_clean",
            ),
            (
                "structured_features",
                MaxAbsScaler(),
                structured_feature_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return Pipeline(
        steps=[
            ("feature_builder", SMSFeatureBuilder()),
            ("feature_union", feature_union),
        ]
    )


class FixedLabelEncoder:
    """Minimal fixed encoder so class IDs stay stable across training runs."""

    def __init__(self, mapping: Dict[str, int]):
        self.mapping = mapping
        self.inverse_mapping = {value: key for key, value in mapping.items()}
        self.classes_ = np.array([label for label, _ in sorted(mapping.items(), key=lambda item: item[1])])

    def transform(self, labels: Iterable[str]) -> np.ndarray:
        return np.array([self.mapping[label] for label in labels], dtype=np.int64)

    def inverse_transform(self, label_ids: Iterable[int]) -> np.ndarray:
        return np.array([self.inverse_mapping[label_id] for label_id in label_ids])


def prepare_train_test_data(
    dataset_path: str = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Pipeline, FixedLabelEncoder]:
    """
    Load data, handle missing values, encode labels, split the dataset,
    and fit the preprocessing pipeline on training data only.
    """
    df = load_sms_dataset(dataset_path)
    X = df[["sms_text", "sender_id"]].copy()
    y = df["target_label"].copy()

    # Encode target labels after standardizing them to the required class names.
    label_encoder = FixedLabelEncoder(LABEL_TO_ID)
    y_encoded = label_encoder.transform(y)

    # Split before fitting TF-IDF to avoid text leakage from test to train.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    preprocessing_pipeline = build_preprocessing_pipeline()
    X_train_features = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_features = preprocessing_pipeline.transform(X_test)

    print("Dataset loaded successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Encoded classes: {list(label_encoder.classes_)}")
    print(f"Train feature matrix shape: {X_train_features.shape}")
    print(f"Test feature matrix shape: {X_test_features.shape}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing_pipeline,
        label_encoder,
    )


def preview_preprocessing(dataset_path: str = DATASET_PATH, sample_rows: int = 5) -> None:
    """Show a small preview of the cleaned and engineered fields for sanity checking."""
    df = load_sms_dataset(dataset_path)
    builder = SMSFeatureBuilder()
    preview = builder.fit_transform(df[["sms_text", "sender_id"]].head(sample_rows))

    columns_to_show = [
        "sms_text",
        "sender_id",
        "clean_text",
        "tokenized_text",
        "message_length",
        "word_count",
        "has_url",
        "has_amount",
        "suspicious_term_count",
    ]
    print(preview[columns_to_show].to_string(index=False))


if __name__ == "__main__":
    preview_preprocessing(DATASET_PATH, sample_rows=5)
    prepare_train_test_data(DATASET_PATH)
