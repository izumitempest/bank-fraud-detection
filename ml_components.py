import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BANK_KEYWORDS = (
    "gtbank",
    "gtb",
    "access bank",
    "access",
    "zenith bank",
    "zenith",
    "uba",
    "firstbank",
    "first bank",
    "stanbic",
    "fidelity",
    "opay",
    "kuda",
    "moniepoint",
    "union bank",
    "wema",
    "sterling",
    "fcmb",
)

URL_PATTERN = re.compile(
    r"((https?://|www\.)\S+|"
    r"\b(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|is\.gd|ow\.ly|buff\.ly|"
    r"[a-z0-9-]+\.(?:xyz|top|click|link|site|live|online|net|com|ng))\S*)",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
AMOUNT_PATTERN = re.compile(
    r"(\b(?:ngn|usd|naira)\b|[#$€£]|₦|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)",
    re.IGNORECASE,
)
UPPERCASE_PATTERN = re.compile(r"[A-Z]")
LETTER_PATTERN = re.compile(r"[A-Za-z]")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
MISSPELLING_PATTERN = re.compile(
    r"\b(acount|recieved|requierd|transacion|regulatons|login here|"
    r"deactivat(?:ed|e)|verfy|clik|blok(?:ed|e)|updte|bvn irregularity|"
    r"otp to receive|kindly send otp|bin restricted)\b",
    re.IGNORECASE,
)

URGENCY_TERMS = (
    "urgent",
    "immediately",
    "now",
    "verify",
    "click",
    "suspended",
    "restricted",
    "blocked",
    "reactivate",
    "deadline",
)
OTP_TERMS = (
    "otp",
    "pin",
    "password",
    "passcode",
    "token",
    "card details",
    "bvn",
    "nin",
    "cvv",
)


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def normalize_sender_id(sender_id: str) -> str:
    sender = _safe_text(sender_id).strip().lower()
    if not sender:
        return "unknown_sender"
    for keyword in BANK_KEYWORDS:
        if keyword in sender:
            return "known_bank"
    return "unknown_sender"


def detect_bank_from_text(text: str) -> str:
    normalized = _safe_text(text).lower()
    for keyword in BANK_KEYWORDS:
        if keyword in normalized:
            return keyword
    return ""


def _keyword_count(text: str, keywords: Iterable[str]) -> int:
    normalized = _safe_text(text).lower()
    return sum(1 for keyword in keywords if keyword in normalized)


class AlertStructuredFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=["text", "sender_id"])

        if "text" not in df.columns:
            raise ValueError("Expected a 'text' column for alert feature extraction.")

        if "sender_id" not in df.columns:
            df["sender_id"] = ""

        rows = []
        for _, row in df.iterrows():
            text = _safe_text(row.get("text"))
            sender_id = _safe_text(row.get("sender_id"))
            letters = LETTER_PATTERN.findall(text)
            uppercase = UPPERCASE_PATTERN.findall(text)
            punctuation = PUNCTUATION_PATTERN.findall(text)
            sender_bank = normalize_sender_id(sender_id)
            mentioned_bank = detect_bank_from_text(text)

            rows.append(
                {
                    "sender_known_bank": 1 if sender_bank == "known_bank" else 0,
                    "sender_unknown": 1 if sender_bank == "unknown_sender" else 0,
                    "sender_matches_bank_mention": 1
                    if sender_bank == "known_bank" and mentioned_bank
                    else 0,
                    "has_url": 1 if URL_PATTERN.search(text) else 0,
                    "has_phone_number": 1 if PHONE_PATTERN.search(text) else 0,
                    "has_amount": 1 if AMOUNT_PATTERN.search(text) else 0,
                    "urgency_keyword_count": _keyword_count(text, URGENCY_TERMS),
                    "otp_keyword_count": _keyword_count(text, OTP_TERMS),
                    "uppercase_ratio": (
                        len(uppercase) / max(len(letters), 1) if letters else 0.0
                    ),
                    "punctuation_ratio": len(punctuation) / max(len(text), 1),
                    "misspelling_indicator": 1 if MISSPELLING_PATTERN.search(text) else 0,
                    "message_length": len(text),
                    "digit_ratio": sum(ch.isdigit() for ch in text) / max(len(text), 1),
                }
            )

        return pd.DataFrame(rows).astype(float)
