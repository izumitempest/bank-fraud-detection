from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import os
import pandas as pd

# Configuration - Alert Classifier
ALERT_MODEL = "/home/izumi/Documents/CODE/Chichi/models/alert_classifier_v2.pkl"
ALERT_VECTORIZER = "/home/izumi/Documents/CODE/Chichi/models/tfidf_vectorizer_v2.pkl"

# Configuration - Fraud Engine (NIBSS)
FRAUD_MODEL = "/home/izumi/Documents/CODE/Chichi/models/fraud_engine_model_v3.pkl"
FRAUD_ENCODERS = "/home/izumi/Documents/CODE/Chichi/models/fraud_engine_encoders.pkl"
FRAUD_FEATURES = "/home/izumi/Documents/CODE/Chichi/models/fraud_engine_features.pkl"
FRAUD_THRESHOLD = 0.20  # Optimized for ~82% recall

app = FastAPI(title="Chichi Fraud Detection API")

# Load alert models
if os.path.exists(ALERT_MODEL) and os.path.exists(ALERT_VECTORIZER):
    alert_clf = joblib.load(ALERT_MODEL)
    alert_vectorizer = joblib.load(ALERT_VECTORIZER)
else:
    alert_clf, alert_vectorizer = None, None

# Load fraud engine models
if (
    os.path.exists(FRAUD_MODEL)
    and os.path.exists(FRAUD_ENCODERS)
    and os.path.exists(FRAUD_FEATURES)
):
    fraud_clf = joblib.load(FRAUD_MODEL)
    fraud_encoders = joblib.load(FRAUD_ENCODERS)
    fraud_features = joblib.load(FRAUD_FEATURES)
else:
    fraud_clf, fraud_encoders, fraud_features = None, None, None

# Models and Map
LABEL_MAP = {0: "Legitimate", 1: "Fake/Phishing", 2: "Suspicious"}


class AlertRequest(BaseModel):
    text: str


class AlertResponse(BaseModel):
    text: str
    prediction: str
    label_id: int


class TransactionRequest(BaseModel):
    # Features required by the fraud engine
    amount: float
    hour: int
    day_of_week: int
    month: int
    is_weekend: bool
    is_peak_hour: bool
    tx_count_24h: float
    amount_sum_24h: float
    amount_mean_7d: float
    amount_std_7d: float
    tx_count_total: int
    amount_mean_total: float
    amount_std_total: float
    channel_diversity: int
    location_diversity: int
    amount_vs_mean_ratio: float
    online_channel_ratio: float
    channel: str
    merchant_category: str
    bank: str


class TransactionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "alert_classifier": "active" if alert_clf else "inactive",
        "fraud_engine": "active" if fraud_clf else "inactive",
    }


@app.post("/predict/alert", response_model=AlertResponse)
def predict_alert(request: AlertRequest):
    if not alert_clf:
        raise HTTPException(status_code=503, detail="Alert classifier not loaded")
    try:
        text_tfidf = alert_vectorizer.transform([request.text])
        prediction_id = int(alert_clf.predict(text_tfidf)[0])
        return AlertResponse(
            text=request.text,
            prediction=LABEL_MAP.get(prediction_id, "Unknown"),
            label_id=prediction_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/transaction", response_model=TransactionResponse)
def predict_transaction(request: TransactionRequest):
    if not fraud_clf:
        raise HTTPException(status_code=503, detail="Fraud engine not loaded")
    try:
        # Convert request to DataFrame
        data = request.dict()
        df = pd.DataFrame([data])

        # Apply encoders
        for col, encoder in fraud_encoders.items():
            # Handle unseen categories by using a fallback or error
            if df[col][0] not in encoder.classes_:
                # For this demo, we'll just use the first class if unseen,
                # in production we'd handle this better
                df[col] = encoder.transform([encoder.classes_[0]])
            else:
                df[col] = encoder.transform(df[col])

        # Ensure feature order matches training
        df = df[fraud_features]

        # Predict Probabilities
        prob = float(fraud_clf.predict_proba(df)[0][1])

        # Apply Tuned Threshold
        is_fraud = prob >= FRAUD_THRESHOLD

        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.2 else "Low"

        return TransactionResponse(
            is_fraud=is_fraud, fraud_probability=prob, risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
