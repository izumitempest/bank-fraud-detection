import joblib
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
MODEL_PATH = os.path.join(BASE_DIR, "models", "alert_classifier_v2.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_v2.pkl")


def predict_alert(text):
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    text_tfidf = vectorizer.transform([text])
    prediction = clf.predict(text_tfidf)[0]

    label_map = {0: "Legitimate", 1: "Fake/Phishing", 2: "Suspicious"}
    return label_map[prediction]


if __name__ == "__main__":
    test_cases = [
        "GTBank ALERT: CREDIT NGN25,000.00 to OLAYINKA BAKARE Ref: 1234567 09-03-2026 Bal: NGN150,000.00",
        "URGENT: Your acount has bin restricted due to central bank regulatons. Pls login here to verify: bit.ly/bank-verify",
        "Security Alert: Your Zenith Bank internet banking password was changed from Lagos. Contact us if not you.",
        "Dear customer, you recieved a reward of NGN10,000 from GTB. Claim within 2hrs: tinyurl.com/bvn-update",
    ]

    print("Model v2 Verification Results:\n")
    for test in test_cases:
        result = predict_alert(test)
        print(f"Text: {test}")
        print(f"Prediction: {result}\n")
