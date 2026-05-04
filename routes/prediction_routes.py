import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("predictions")
router = APIRouter(prefix="/predict", tags=["predictions"])


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


class BatchAlertRequest(BaseModel):
    alerts: list[AlertRequest]


class BatchAlertResponse(BaseModel):
    results: list[AlertResponse]
    total: int


def _get_sms_artifacts():
    """Get SMS model artifacts from global state."""
    from app import sms_model_bundle
    
    if not isinstance(sms_model_bundle, dict):
        return None, None, None

    preprocessor = sms_model_bundle.get("preprocessor")
    model = sms_model_bundle.get("model")
    label_encoder = sms_model_bundle.get("label_encoder")
    return preprocessor, model, label_encoder


def _get_fraud_pipeline():
    """Get fraud detection pipeline from global state."""
    from app import fraud_pipeline
    return fraud_pipeline


@router.post("/alert", response_model=AlertResponse)
def predict_alert(request: AlertRequest):
    """Predict SMS fraud classification: Legitimate, Fake/Phishing, or Suspicious."""
    preprocessor, sms_model, label_encoder = _get_sms_artifacts()
    if not preprocessor or not sms_model or not label_encoder:
        raise HTTPException(status_code=503, detail="SMS fraud model is not loaded")

    try:
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

        logger.info(
            f"SMS prediction: text_len={len(request.text)} prediction={prediction_label} confidence={confidence:.3f}"
        )

        return AlertResponse(
            text=request.text,
            sender_id=request.sender_id,
            prediction=prediction_label,
            label_id=prediction_id,
            confidence=confidence,
            probabilities=probability_map,
        )
    except Exception as exc:
        logger.error(f"SMS prediction error: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/alert/batch", response_model=BatchAlertResponse)
def predict_alerts_batch(request: BatchAlertRequest):
    """Batch predict SMS fraud classification for multiple alerts."""
    preprocessor, sms_model, label_encoder = _get_sms_artifacts()
    if not preprocessor or not sms_model or not label_encoder:
        raise HTTPException(status_code=503, detail="SMS fraud model is not loaded")

    if not request.alerts or len(request.alerts) > 100:
        raise HTTPException(
            status_code=400, detail="Batch size must be between 1 and 100"
        )

    try:
        results = []
        for alert_request in request.alerts:
            payload = pd.DataFrame(
                [{"sms_text": alert_request.text, "sender_id": alert_request.sender_id or ""}]
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

            results.append(
                AlertResponse(
                    text=alert_request.text,
                    sender_id=alert_request.sender_id,
                    prediction=prediction_label,
                    label_id=prediction_id,
                    confidence=confidence,
                    probabilities=probability_map,
                )
            )

        logger.info(f"Batch SMS predictions: count={len(results)}")
        return BatchAlertResponse(results=results, total=len(results))
    except Exception as exc:
        logger.error(f"Batch SMS prediction error: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/transaction", response_model=TransactionResponse)
def predict_transaction(request: TransactionRequest):
    """Predict transaction fraud with risk level classification."""
    fraud_pipeline = _get_fraud_pipeline()
    if fraud_pipeline is None:
        raise HTTPException(status_code=503, detail="Fraud engine not loaded")

    try:
        payload = pd.DataFrame([request.dict()])
        prediction = int(fraud_pipeline.predict(payload)[0])
        probabilities = fraud_pipeline.predict_proba(payload)[0]
        fraud_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        risk_level = (
            "High"
            if fraud_probability >= 0.8
            else "Medium"
            if fraud_probability >= 0.5
            else "Low"
        )

        logger.info(
            f"Transaction prediction: is_fraud={bool(prediction)} risk={risk_level} confidence={confidence:.3f}"
        )

        return TransactionResponse(
            is_fraud=bool(prediction),
            fraud_probability=fraud_probability,
            confidence=confidence,
            risk_level=risk_level,
        )
    except Exception as exc:
        logger.error(f"Transaction prediction error: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))
