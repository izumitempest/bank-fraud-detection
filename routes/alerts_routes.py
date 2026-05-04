import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("alerts")
router = APIRouter(prefix="/api/alerts", tags=["alerts"])


class AlertHistoryResponse(BaseModel):
    id: str
    text: str
    sender_id: Optional[str] = None
    prediction: str
    confidence: float
    created_at: str


class AlertsHistoryResponse(BaseModel):
    alerts: list[AlertHistoryResponse]
    total: int


# In-memory alert history (consider moving to database for production)
_alert_history = []


def record_alert(alert_id: str, text: str, sender_id: Optional[str], prediction: str, confidence: float, created_at: str) -> None:
    """Record SMS alert for history tracking."""
    _alert_history.append(
        {
            "id": alert_id,
            "text": text,
            "sender_id": sender_id,
            "prediction": prediction,
            "confidence": confidence,
            "created_at": created_at,
        }
    )


@router.get("/", response_model=AlertsHistoryResponse)
def get_alerts_history(limit: int = 50, offset: int = 0):
    """Get alert history with pagination."""
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400, detail="Limit must be between 1 and 500"
        )
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be >= 0")

    paginated = _alert_history[offset : offset + limit]
    
    logger.info(f"Alerts history requested: limit={limit}, offset={offset}, returned={len(paginated)}")

    return AlertsHistoryResponse(
        alerts=[AlertHistoryResponse(**alert) for alert in paginated],
        total=len(_alert_history),
    )


@router.get("/{alert_id}", response_model=AlertHistoryResponse)
def get_alert_detail(alert_id: str):
    """Get details of a specific alert by ID."""
    for alert in _alert_history:
        if alert["id"] == alert_id:
            logger.info(f"Alert detail requested: id={alert_id}")
            return AlertHistoryResponse(**alert)

    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@router.get("/sender/{sender_id}", response_model=AlertsHistoryResponse)
def get_alerts_by_sender(sender_id: str, limit: int = 50):
    """Get all alerts from a specific sender."""
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400, detail="Limit must be between 1 and 500"
        )

    filtered = [alert for alert in _alert_history if alert["sender_id"] == sender_id]
    paginated = filtered[:limit]

    logger.info(f"Alerts by sender requested: sender_id={sender_id}, returned={len(paginated)}")

    return AlertsHistoryResponse(
        alerts=[AlertHistoryResponse(**alert) for alert in paginated],
        total=len(filtered),
    )
