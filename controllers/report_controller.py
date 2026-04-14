from datetime import datetime, timezone

from fastapi import HTTPException

from config.supabase import get_supabase
from models.report import ReportCreate, ReportResponse


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

    response = supabase.from_("reports").insert(payload).execute()
    if getattr(response, "error", None):
        raise HTTPException(status_code=500, detail=response.error.message)
    if not response.data:
        raise HTTPException(
            status_code=500, detail="Supabase did not return a saved report record."
        )

    saved_report = response.data[0]
    return {
        "success": True,
        "report_id": str(saved_report.get("id", "")),
        "message": "Report saved successfully.",
        "data": saved_report,
    }


def list_reports() -> list[ReportResponse]:
    supabase = get_supabase()
    response = (
        supabase.from_("reports").select("*").order("created_at", desc=True).execute()
    )
    if getattr(response, "error", None):
        raise HTTPException(status_code=500, detail=response.error.message)

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
