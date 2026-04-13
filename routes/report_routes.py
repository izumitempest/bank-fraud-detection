import logging

from fastapi import APIRouter

from controllers.report_controller import create_report, list_reports
from models.report import ReportCreate, ReportResponse


router = APIRouter(tags=["reports"])
logger = logging.getLogger("reports")


@router.post("/report")
def submit_report(report: ReportCreate):
    logger.info(
        "Incoming report payload: message=%s predicted_label=%s corrected_label=%s confidence=%s sender=%s",
        report.message,
        report.predicted_label,
        report.corrected_label,
        report.confidence,
        report.sender,
    )
    return create_report(report)


@router.get("/reports", response_model=list[ReportResponse])
def get_reports():
    return list_reports()
