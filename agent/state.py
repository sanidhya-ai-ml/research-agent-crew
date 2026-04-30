from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict):
    task_id: str
    query: str
    plan: list[str]            # sub-questions produced by planner
    research_results: dict     # sub-question → search findings string
    summary: str               # condensed findings from summariser
    fact_check_notes: str      # confidence flags from fact_checker
    draft: str                 # polished draft from editor
    human_feedback: str        # rejection feedback, injected before revision
    revision_count: int        # incremented on each rejection
    approved: bool             # set via update_state before resuming
    final_report: str          # output from finaliser
