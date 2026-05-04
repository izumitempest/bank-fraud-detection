import logging

from fastapi import APIRouter, HTTPException

from controllers.report_controller import create_report, list_reports
from models.report import ReportCreate, ReportResponse


router = APIRouter(tags=["reports"])
logger = logging.getLogger("reports")


@router.post("/report")
def submit_report(report: ReportCreate):
    logger.info(
        "Incoming report: message_len=%d predicted_label=%s corrected_label=%s confidence=%s sender=%s",
        len(report.message),
        report.predicted_label,
        report.corrected_label,
        report.confidence,
        report.sender,
    )
    try:
        return create_report(report)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in submit_report: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@router.get("/reports", response_model=list[ReportResponse])
def get_reports():
    return list_reports()
