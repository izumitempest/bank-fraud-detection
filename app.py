from typing import Optional
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from config.supabase import (
        close_supabase_connection,
        connect_to_supabase,
        is_supabase_ready,
    )
except Exception:
    # Keep the prediction API available even if the optional database
    # dependency is not installed in the runtime environment.
    def connect_to_supabase():
        return None

    def close_supabase_connection():
        return None

    def is_supabase_ready():
        return False

try:
    from routes.report_routes import router as report_router
except Exception:
    report_router = None

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
SMS_MODEL_BUNDLE = os.path.join(BASE_DIR, "models", "sms_xgboost.pkl")
FRAUD_MODEL = os.path.join(BASE_DIR, "models", "fraud_engine_model_v3.pkl")

app = FastAPI(title="Chichi Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_pipeline(path: str):
    return joblib.load(path) if os.path.exists(path) else None


# The SMS bundle contains:
# - preprocessor: text cleaning + TF-IDF + engineered features
# - model: trained XGBoost classifier
# - label_encoder: converts class IDs back to Real/Fake/Suspicious
sms_model_bundle = _load_pipeline(SMS_MODEL_BUNDLE)
fraud_pipeline = _load_pipeline(FRAUD_MODEL)


def _get_sms_artifacts():
    if not isinstance(sms_model_bundle, dict):
        return None, None, None

    preprocessor = sms_model_bundle.get("preprocessor")
    model = sms_model_bundle.get("model")
    label_encoder = sms_model_bundle.get("label_encoder")
    return preprocessor, model, label_encoder


@app.on_event("startup")
def startup_event():
    connect_to_supabase()


@app.on_event("shutdown")
def shutdown_event():
    close_supabase_connection()


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


class AlertRequest(BaseModel):
    text: str
    sender_id: Optional[str] = None


class AlertResponse(BaseModel):
    text: str
    sender_id: Optional[str] = None
    prediction: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]


class TransactionRequest(BaseModel):
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
    confidence: float
    risk_level: str


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Chimera Fraud Detection API",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "sms_fraud_model": "active" if sms_model_bundle else "inactive",
        "fraud_engine": "active" if fraud_pipeline else "inactive",
        "database": "connected" if is_supabase_ready() else "unavailable",
    }


@app.post("/predict/alert", response_model=AlertResponse)
def predict_alert(request: AlertRequest):
    preprocessor, sms_model, label_encoder = _get_sms_artifacts()
    if not preprocessor or not sms_model or not label_encoder:
        raise HTTPException(status_code=503, detail="SMS fraud model is not loaded")

    try:
        # Build a one-row DataFrame so the saved preprocessing pipeline can
        # apply the same cleaning and feature engineering used during training.
        payload = pd.DataFrame(
            [{"sms_text": request.text, "sender_id": request.sender_id or ""}]
        )

        features = preprocessor.transform(payload)
        prediction_id = int(sms_model.predict(features)[0])
        probabilities = sms_model.predict_proba(features)[0]
        confidence = float(probabilities[prediction_id])
        prediction_label = str(label_encoder.inverse_transform([prediction_id])[0])
        probability_map = {
            str(class_name): float(probability)
            for class_name, probability in zip(label_encoder.classes_, probabilities)
        }

        return AlertResponse(
            text=request.text,
            sender_id=request.sender_id,
            prediction=prediction_label,
            label_id=prediction_id,
            confidence=confidence,
            probabilities=probability_map,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/transaction", response_model=TransactionResponse)
def predict_transaction(request: TransactionRequest):
    if fraud_pipeline is None:
        raise HTTPException(status_code=503, detail="Fraud engine not loaded")

    try:
        payload = pd.DataFrame([request.dict()])
        prediction = int(fraud_pipeline.predict(payload)[0])
        probabilities = fraud_pipeline.predict_proba(payload)[0]
        fraud_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        risk_level = "High" if fraud_probability >= 0.8 else "Medium" if fraud_probability >= 0.5 else "Low"

        return TransactionResponse(
            is_fraud=bool(prediction),
            fraud_probability=fraud_probability,
            confidence=confidence,
            risk_level=risk_level,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if report_router is not None:
    app.include_router(report_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
