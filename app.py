import logging
import os

import joblib
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Supabase utilities (optional)
try:
    from config.supabase import (
        close_supabase_connection,
        connect_to_supabase,
        is_supabase_ready,
    )
except Exception as e:
    logger.warning(f"Supabase config not available: {e}")

    def connect_to_supabase():
        return None

    def close_supabase_connection():
        return None

    def is_supabase_ready():
        return False


# Try to import routers
try:
    from routes.prediction_routes import router as prediction_router
except Exception as e:
    logger.error(f"Failed to import prediction_router: {e}")
    prediction_router = None

try:
    from routes.health_routes import router as health_router
except Exception as e:
    logger.error(f"Failed to import health_router: {e}")
    health_router = None

try:
    from routes.report_routes import router as report_router
except Exception as e:
    logger.warning(f"Report router not available: {e}")
    report_router = None

try:
    from routes.analytics_routes import router as analytics_router
except Exception as e:
    logger.warning(f"Analytics router not available: {e}")
    analytics_router = None

try:
    from routes.alerts_routes import router as alerts_router
except Exception as e:
    logger.warning(f"Alerts router not available: {e}")
    alerts_router = None

# Base directory and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMS_MODEL_BUNDLE = os.path.join(BASE_DIR, "models", "sms_xgboost.pkl")
FRAUD_MODEL = os.path.join(BASE_DIR, "models", "fraud_engine_model_v3.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="Chichi Fraud Detection API",
    description="Advanced fraud detection for SMS alerts and transactions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_pipeline(path: str):
    """Load a trained pipeline from disk."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load pipeline {path}: {e}")
            return None
    return None


# Load models at startup
sms_model_bundle = _load_pipeline(SMS_MODEL_BUNDLE)
fraud_pipeline = _load_pipeline(FRAUD_MODEL)

logger.info(f"SMS model loaded: {sms_model_bundle is not None}")
logger.info(f"Fraud engine loaded: {fraud_pipeline is not None}")


@app.on_event("startup")
def startup_event():
    """Initialize connections on app startup."""
    connect_to_supabase()
    logger.info("Application startup complete")


@app.on_event("shutdown")
def shutdown_event():
    """Clean up connections on app shutdown."""
    close_supabase_connection()
    logger.info("Application shutdown complete")


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request, exc: RequestValidationError):
    """Custom handler for validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Include routers
if health_router:
    app.include_router(health_router)
    logger.info("Health router included")

if prediction_router:
    app.include_router(prediction_router)
    logger.info("Prediction router included")

if report_router:
    app.include_router(report_router)
    logger.info("Report router included")

if analytics_router:
    app.include_router(analytics_router)
    logger.info("Analytics router included")

if alerts_router:
    app.include_router(alerts_router)
    logger.info("Alerts router included")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
