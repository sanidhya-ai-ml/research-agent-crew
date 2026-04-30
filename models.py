from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500, description="Research question or topic")


class FeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=1, max_length=2000, description="Rejection feedback for editor")


class TaskResponse(BaseModel):
    task_id: str
    status: str
    query: str = ""
    progress: str = ""
    created_at: str = ""


class DraftResponse(BaseModel):
    task_id: str
    status: str
    draft: str
    revision_count: int = 0
    message: str = "Review the draft and approve or reject with feedback."


class ReportResponse(BaseModel):
    task_id: str
    status: str
    report: str
    query: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    redis: bool = False
