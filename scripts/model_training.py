import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Configuration
DATASET_PATH = "/home/izumi/Documents/CODE/Chichi/data/alerts_dataset_v2.csv"
MODEL_PATH = "/home/izumi/Documents/CODE/Chichi/models/alert_classifier_v2.pkl"
VECTORIZER_PATH = "/home/izumi/Documents/CODE/Chichi/models/tfidf_vectorizer_v2.pkl"


def train_model():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    # Load dataset
    df = pd.read_csv(DATASET_FILE)
    X = df["text"]
    y = df["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorize text
    # Using ngram_range=(1, 2) to capture skip-gram patterns like "account blocked"
    vectorizer = TfidfVectorizer(
        stop_words="english", lowercase=True, ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model (Random Forest for better robustness)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    print("Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Legitimate", "Fake", "Suspicious"]
        )
    )

    # Save model and vectorizer
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Vectorizer saved to {VECTORIZER_FILE}")


if __name__ == "__main__":
    train_model()
