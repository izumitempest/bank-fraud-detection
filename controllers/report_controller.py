import logging
from datetime import datetime, timezone

from fastapi import HTTPException

from config.supabase import get_supabase
from models.report import ReportCreate, ReportResponse

logger = logging.getLogger("report_controller")


def create_report(report: ReportCreate) -> dict:
    supabase = get_supabase()
    payload = {
        "message": report.message,
        "predicted_label": report.predicted_label,
        "corrected_label": report.corrected_label,
        "confidence": report.confidence,
        "sender": report.sender,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = supabase.from_("reports").insert(payload).execute()
    except Exception as exc:
        logger.error("Supabase insert failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not response.data:
        logger.error("Supabase insert returned no data; payload=%s", payload)
        raise HTTPException(
            status_code=500, detail="Supabase did not return a saved report record."
        )

    saved_report = response.data[0]
    logger.info("Report saved: id=%s", saved_report.get("id"))
    return {
        "success": True,
        "report_id": str(saved_report.get("id", "")),
        "message": "Report saved successfully.",
        "data": saved_report,
    }


def list_reports() -> list[ReportResponse]:
    supabase = get_supabase()
    try:
        response = (
            supabase.from_("reports").select("*").order("created_at", desc=True).execute()
        )
    except Exception as exc:
        logger.error("Supabase select failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    reports: list[ReportResponse] = []
    for document in response.data or []:
        reports.append(
            ReportResponse(
                id=str(document.get("id", "")),
                message=document["message"],
                predicted_label=document["predicted_label"],
                corrected_label=document["corrected_label"],
                confidence=document.get("confidence"),
                sender=document.get("sender"),
                created_at=document["created_at"],
            )
        )

    return reports
