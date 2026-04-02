from fastapi import APIRouter

from controllers.report_controller import create_report, list_reports
from models.report import ReportCreate, ReportResponse


router = APIRouter(tags=["reports"])


@router.post("/report")
def submit_report(report: ReportCreate):
    return create_report(report)


@router.get("/reports", response_model=list[ReportResponse])
def get_reports():
    return list_reports()
