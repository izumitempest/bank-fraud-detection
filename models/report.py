from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

class ReportCreate(BaseModel):
    message: str = Field(..., min_length=1)
    predicted_label: str = Field(..., min_length=1)
    corrected_label: str = Field(..., min_length=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sender: Optional[str] = None


class ReportInDB(ReportCreate):
    created_at: datetime


class ReportResponse(ReportInDB):
    id: str
