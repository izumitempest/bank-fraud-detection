import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("health")
router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    sms_fraud_model: str
    fraud_engine: str
    database: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    description: str
    input_fields: list[str]
    output_type: str


class ModelsInfoResponse(BaseModel):
    models: list[ModelInfoResponse]
    api_version: str


@router.get("/", tags=["root"])
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Chichi Fraud Detection API",
        "documentation": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predictions": "/predict/*",
            "reports": "/report*",
            "analytics": "/api/analytics/*",
            "models": "/api/models/info",
        },
    }


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and dependency health status."""
    from app import sms_model_bundle, fraud_pipeline, is_supabase_ready

    return HealthResponse(
        status="healthy",
        sms_fraud_model="active" if sms_model_bundle else "inactive",
        fraud_engine="active" if fraud_pipeline else "inactive",
        database="connected" if is_supabase_ready() else "unavailable",
    )


@router.get("/api/models/info", response_model=ModelsInfoResponse)
def get_models_info():
    """Get information about available prediction models."""
    from app import sms_model_bundle, fraud_pipeline

    models = []

    if sms_model_bundle:
        models.append(
            ModelInfoResponse(
                model_name="SMS Fraud Detection",
                version="1.0",
                description="Classifies SMS messages as Legitimate, Fake/Phishing, or Suspicious",
                input_fields=["text", "sender_id"],
                output_type="categorical (3 classes)",
            )
        )

    if fraud_pipeline:
        models.append(
            ModelInfoResponse(
                model_name="Transaction Fraud Detection",
                version="3.0",
                description="Predicts transaction fraud with risk level assessment",
                input_fields=[
                    "amount",
                    "hour",
                    "day_of_week",
                    "month",
                    "is_weekend",
                    "is_peak_hour",
                    "tx_count_24h",
                    "amount_sum_24h",
                    "amount_mean_7d",
                    "amount_std_7d",
                    "tx_count_total",
                    "amount_mean_total",
                    "amount_std_total",
                    "channel_diversity",
                    "location_diversity",
                    "amount_vs_mean_ratio",
                    "online_channel_ratio",
                    "channel",
                    "merchant_category",
                    "bank",
                ],
                output_type="binary + risk_level",
            )
        )

    logger.info(f"Models info requested: {len(models)} models available")

    return ModelsInfoResponse(models=models, api_version="1.0")
