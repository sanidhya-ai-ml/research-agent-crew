from __future__ import annotations

import logging
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.state import ResearchState
from agent.nodes import (
    editor_node,
    fact_checker_node,
    finaliser_node,
    human_review_node,
    planner_node,
    researcher_node,
    summariser_node,
)

logger = logging.getLogger(__name__)


def _route_after_review(state: ResearchState) -> str:
    """Route from human_review: approved → finaliser, else back to editor (up to MAX_REVISIONS)."""
    if state.get("approved", False):
        return "finaliser"
    max_rev = int(os.environ.get("MAX_REVISIONS", "3"))
    if state.get("revision_count", 0) >= max_rev:
        logger.warning("Max revisions (%d) reached — forcing finalise", max_rev)
        return "finaliser"
    return "editor"


def build_graph() -> StateGraph:
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("summariser", summariser_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("editor", editor_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finaliser", finaliser_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "summariser")
    builder.add_edge("summariser", "fact_checker")
    builder.add_edge("fact_checker", "editor")
    builder.add_edge("editor", "human_review")
    builder.add_conditional_edges(
        "human_review",
        _route_after_review,
        {"finaliser": "finaliser", "editor": "editor"},
    )
    builder.add_edge("finaliser", END)

    return builder


# Compiled singleton — interrupt_before pauses graph just before human_review runs
_checkpointer = MemorySaver()
graph = build_graph().compile(
    checkpointer=_checkpointer,
    interrupt_before=["human_review"],
)
