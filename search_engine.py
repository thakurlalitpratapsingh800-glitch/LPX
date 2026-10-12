"""Deprecated shim.

The original single-question engine lived here and optionally called a paid
OpenAI-compatible API. Askwise is now 100% free, so that logic moved to
`askwise_api.search_engine` (free engines + fallback) and `askwise_api.rag`
(answer pipeline with local model support).

Kept only so old imports don't break the legacy /api/search endpoint.
"""
from askwise_api.search_engine import SearchError, extractive_answer, web_search  # noqa: F401


def answer_question(question: str):
    results, _engine = web_search(question)
    return {
        "answer": extractive_answer(question, results),
        "sources": results,
        "mode": "search",
    }
