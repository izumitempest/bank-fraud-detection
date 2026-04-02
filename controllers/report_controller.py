from datetime import datetime, timezone

from config.db import get_database
from models.report import ReportCreate, ReportResponse


def create_report(report: ReportCreate) -> dict:
    database = get_database()
    payload = {
        "message": report.message,
        "predicted_label": report.predicted_label,
        "corrected_label": report.corrected_label,
        "confidence": report.confidence,
        "sender": report.sender,
        "created_at": datetime.now(timezone.utc),
    }

    result = database.reports.insert_one(payload)
    return {
        "success": True,
        "report_id": str(result.inserted_id),
        "message": "Report saved successfully.",
    }


def list_reports() -> list[ReportResponse]:
    database = get_database()
    reports = []

    for document in database.reports.find().sort("created_at", -1):
        reports.append(
            ReportResponse(
                id=str(document["_id"]),
                message=document["message"],
                predicted_label=document["predicted_label"],
                corrected_label=document["corrected_label"],
                confidence=document.get("confidence"),
                sender=document.get("sender"),
                created_at=document["created_at"],
            )
        )

    return reports
