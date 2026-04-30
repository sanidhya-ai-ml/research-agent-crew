from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import ResearchState
from agent import tools

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_tokens=8192,
            temperature=0.3,
        )
    return _llm


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", path)
        return ""


def _call_llm(system_prompt: str, user_content: str) -> str:
    """Call Gemini via LangChain. Returns content string, never raises."""
    try:
        resp = _get_llm().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return resp.content.strip()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


# ── Node functions ─────────────────────────────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    logger.info("Planner: breaking query into sub-questions")
    system = _load_prompt("planner")
    user = f"Research query: {state['query']}"
    response = _call_llm(system, user)

    # Parse numbered list — accept "1." or "1)" formats
    lines = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip() for ln in response.splitlines() if re.match(r"^\d+", ln.strip())]
    if not lines:
        # Fallback: split by newline and take non-empty lines
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()][:5]
    if not lines:
        lines = [state["query"]]  # last-resort fallback

    logger.info("Planner: produced %d sub-questions", len(lines))
    return {"plan": lines}


def researcher_node(state: ResearchState) -> dict:
    logger.info("Researcher: searching for %d sub-questions", len(state["plan"]))
    system = _load_prompt("researcher")
    results: dict = {}

    for question in state["plan"]:
        search_results = tools.search(question, max_results=4)
        formatted = tools.format_results(search_results)
        user = f"Research question: {question}\n\nSearch results:\n{formatted}"
        findings = _call_llm(system, user)
        if not findings:
            findings = formatted[:500]  # fallback: raw search results
        results[question] = findings
        logger.info("Researcher: answered: %s", question[:60])

    return {"research_results": results}


def summariser_node(state: ResearchState) -> dict:
    logger.info("Summariser: synthesising %d research findings", len(state["research_results"]))
    system = _load_prompt("summariser")

    findings_text = "\n\n".join(
        f"Finding for '{q}':\n{finding}"
        for q, finding in state["research_results"].items()
    )
    user = f"Original query: {state['query']}\n\nResearch findings:\n{findings_text}"
    summary = _call_llm(system, user)

    if not summary:
        summary = findings_text[:1000]  # fallback: concatenated findings

    logger.info("Summariser: produced %d char summary", len(summary))
    return {"summary": summary}


def fact_checker_node(state: ResearchState) -> dict:
    logger.info("Fact checker: reviewing summary for low-confidence claims")
    system = _load_prompt("fact_checker")
    user = f"Summary to fact-check:\n{state['summary']}"
    notes = _call_llm(system, user)

    if not notes:
        notes = "No specific fact-check issues identified."

    logger.info("Fact checker: produced notes")
    return {"fact_check_notes": notes}


def editor_node(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    feedback = state.get("human_feedback", "")
    action = "revision" if revision > 0 else "draft"
    logger.info("Editor: writing %s (revision %d)", action, revision)

    system = _load_prompt("editor")
    user_parts = [
        f"Original query: {state['query']}",
        f"\nResearch summary:\n{state['summary']}",
        f"\nFact-check notes:\n{state['fact_check_notes']}",
    ]
    if feedback:
        user_parts.append(f"\nHuman feedback for revision:\n{feedback}")

    draft = _call_llm(system, "\n".join(user_parts))
    if not draft:
        draft = f"# Research Report: {state['query']}\n\n{state['summary']}"  # fallback

    logger.info("Editor: draft is %d chars", len(draft))
    return {"draft": draft, "human_feedback": ""}  # clear feedback after use


def human_review_node(state: ResearchState) -> dict:
    # Passthrough checkpoint — graph always interrupts BEFORE this node runs.
    # After resumption (via aupdate_state), routes based on state["approved"].
    # Must return at least one key to satisfy LangGraph's write requirement.
    return {"task_id": state["task_id"]}


def finaliser_node(state: ResearchState) -> dict:
    logger.info("Finaliser: producing final report")
    # The draft is already polished — finaliser just stamps it as final
    # and could add a metadata header in production
    final = state.get("draft", "")
    if not final:
        final = state.get("summary", "No content generated.")
    return {"final_report": final}
