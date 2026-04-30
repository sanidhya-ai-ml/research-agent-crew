from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MOCK_RESULTS = [
    {
        "title": "Recent Advances in AI Research",
        "content": "Large language models have shown significant improvements in reasoning and planning capabilities. Multi-agent systems are becoming increasingly practical for complex tasks.",
        "url": "https://example.com/ai-research",
    },
    {
        "title": "State of the Art in Machine Learning",
        "content": "Transformer architectures continue to dominate natural language processing. Retrieval-augmented generation (RAG) has improved factual accuracy. Agent frameworks enable autonomous task completion.",
        "url": "https://example.com/ml-sota",
    },
    {
        "title": "Industry Applications and Trends",
        "content": "Enterprise adoption of AI tools accelerated in 2024. Key applications include code generation, document analysis, customer support automation, and research assistance.",
        "url": "https://example.com/ai-industry",
    },
]


def search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily. Falls back to mock results if API key not set."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — using mock search results for: %s", query)
        return _MOCK_RESULTS[:max_results]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        return response.get("results", [])
    except Exception as exc:
        logger.warning("Tavily search failed (%s) — using mock results", exc)
        return _MOCK_RESULTS[:max_results]


def format_results(results: list[dict]) -> str:
    """Format search results into a readable text block."""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        content = r.get("content", r.get("snippet", ""))
        url = r.get("url", "")
        lines.append(f"[{i}] {title}\n{content}\nSource: {url}")
    return "\n\n".join(lines)
