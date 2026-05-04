import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("analytics")
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    sms_predictions: int
    transaction_predictions: int
    average_sms_confidence: float
    average_transaction_confidence: float
    fraud_detection_rate: float
    last_prediction_time: Optional[str] = None


class ReportStatsResponse(BaseModel):
    total_reports: int
    reports_by_label: dict[str, int]
    average_confidence_reported: float
    most_common_correction: Optional[str] = None
    last_report_time: Optional[str] = None


class ModelPerformanceResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[dict] = None


# In-memory stats (consider moving to database for production)
_prediction_stats = {
    "total_predictions": 0,
    "sms_predictions": 0,
    "transaction_predictions": 0,
    "sms_confidence_sum": 0.0,
    "transaction_confidence_sum": 0.0,
    "fraud_count": 0,
    "last_prediction_time": None,
}

_report_stats = {
    "total_reports": 0,
    "real_count": 0,
    "fake_count": 0,
    "suspicious_count": 0,
    "confidence_sum": 0.0,
    "last_report_time": None,
}


def record_sms_prediction(confidence: float) -> None:
    """Record SMS prediction for analytics."""
    _prediction_stats["sms_predictions"] += 1
    _prediction_stats["total_predictions"] += 1
    _prediction_stats["sms_confidence_sum"] += confidence
    _prediction_stats["last_prediction_time"] = datetime.utcnow().isoformat()


def record_transaction_prediction(confidence: float, is_fraud: bool) -> None:
    """Record transaction prediction for analytics."""
    _prediction_stats["transaction_predictions"] += 1
    _prediction_stats["total_predictions"] += 1
    _prediction_stats["transaction_confidence_sum"] += confidence
    if is_fraud:
        _prediction_stats["fraud_count"] += 1
    _prediction_stats["last_prediction_time"] = datetime.utcnow().isoformat()


def record_report(label: str, confidence: float) -> None:
    """Record user report for analytics."""
    _report_stats["total_reports"] += 1
    _report_stats["confidence_sum"] += confidence
    _report_stats["last_report_time"] = datetime.utcnow().isoformat()

    if label == "Real":
        _report_stats["real_count"] += 1
    elif label == "Fake":
        _report_stats["fake_count"] += 1
    elif label == "Suspicious":
        _report_stats["suspicious_count"] += 1


@router.get("/predictions", response_model=PredictionStatsResponse)
def get_prediction_stats():
    """Get prediction statistics and model performance metrics."""
    avg_sms_conf = (
        _prediction_stats["sms_confidence_sum"] / _prediction_stats["sms_predictions"]
        if _prediction_stats["sms_predictions"] > 0
        else 0.0
    )
    avg_tx_conf = (
        _prediction_stats["transaction_confidence_sum"]
        / _prediction_stats["transaction_predictions"]
        if _prediction_stats["transaction_predictions"] > 0
        else 0.0
    )
    fraud_rate = (
        _prediction_stats["fraud_count"] / _prediction_stats["transaction_predictions"]
        if _prediction_stats["transaction_predictions"] > 0
        else 0.0
    )

    logger.info("Prediction stats requested")

    return PredictionStatsResponse(
        total_predictions=_prediction_stats["total_predictions"],
        sms_predictions=_prediction_stats["sms_predictions"],
        transaction_predictions=_prediction_stats["transaction_predictions"],
        average_sms_confidence=round(avg_sms_conf, 4),
        average_transaction_confidence=round(avg_tx_conf, 4),
        fraud_detection_rate=round(fraud_rate, 4),
        last_prediction_time=_prediction_stats["last_prediction_time"],
    )


@router.get("/reports", response_model=ReportStatsResponse)
def get_report_stats():
    """Get user report statistics and feedback analysis."""
    avg_confidence = (
        _report_stats["confidence_sum"] / _report_stats["total_reports"]
        if _report_stats["total_reports"] > 0
        else 0.0
    )

    # Determine most common correction
    most_common = None
    if _report_stats["total_reports"] > 0:
        counts = {
            "Real": _report_stats["real_count"],
            "Fake": _report_stats["fake_count"],
            "Suspicious": _report_stats["suspicious_count"],
        }
        most_common = max(counts, key=counts.get)

    logger.info("Report stats requested")

    return ReportStatsResponse(
        total_reports=_report_stats["total_reports"],
        reports_by_label={
            "Real": _report_stats["real_count"],
            "Fake": _report_stats["fake_count"],
            "Suspicious": _report_stats["suspicious_count"],
        },
        average_confidence_reported=round(avg_confidence, 4),
        most_common_correction=most_common,
        last_report_time=_report_stats["last_report_time"],
    )


@router.get("/performance")
def get_model_performance():
    """Get detailed model performance metrics (placeholder for future database integration)."""
    logger.info("Model performance requested")
    return {
        "message": "Model performance metrics available via evaluation scripts",
        "evaluation_path": "reports/evaluation_charts/",
        "note": "Integrate with Supabase to store metrics over time",
    }


@router.get("/daily-summary")
def get_daily_summary(days: int = 7):
    """Get summary statistics for the last N days."""
    logger.info(f"Daily summary requested for last {days} days")
    return {
        "period_days": days,
        "total_predictions": _prediction_stats["total_predictions"],
        "average_daily_predictions": round(
            _prediction_stats["total_predictions"] / max(days, 1), 2
        ),
        "fraud_cases": _prediction_stats["fraud_count"],
        "user_reports": _report_stats["total_reports"],
        "note": "Time-series data requires database integration",
    }
