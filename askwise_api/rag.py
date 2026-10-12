
"""Askwise answer pipeline: local Ollama, conversation memory, PDFs and web search."""

from __future__ import annotations

import logging
import re
from typing import Any

from . import database
from .ollama_client import chat as ollama_chat
from .ollama_client import model_available
from .search_engine import SearchError, extractive_answer, web_search

logger = logging.getLogger(__name__)

MAX_SNIPPET_CHARS = 850
MAX_WEB_RESULTS = 5

SYSTEM_PROMPT = """
You are LPX, a helpful AI assistant.
Answer the user's current question directly.
Use simple, natural English unless another language is requested.
For general knowledge and simple maths, use your own knowledge.
Explain programming concepts with examples when useful.
Use supplied documents and web evidence only when relevant.
Never invent facts, citations, or URLs.
Do not include a Sources section unless explicitly requested.
Do not answer an unrelated previous question.
If uncertain, state the uncertainty clearly.
"""


def _clean_evidence(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:MAX_SNIPPET_CHARS]


def _build_context(
    question: str,
    conversation_id: int | None,
    web_results: list[dict[str, str]] | None,
    docs: list[dict] | None,
) -> str:
    parts: list[str] = []

    if docs:
        chunks = []
        for doc in docs:
            filename = str(doc.get("filename", "Uploaded document"))
            content = _clean_evidence(str(doc.get("text", "")))
            if content:
                chunks.append(f"Document: {filename}\n{content[:3000]}")
        if chunks:
            parts.append("[Uploaded documents]\n" + "\n\n".join(chunks))

    if web_results:
        sources = []
        for i, result in enumerate(web_results[:MAX_WEB_RESULTS], start=1):
            title = _clean_evidence(result.get("title", "Untitled source"))
            snippet = _clean_evidence(result.get("snippet", ""))
            if snippet:
                sources.append(f"Source [{i}]: {title}\nEvidence: {snippet}")
        if sources:
            parts.append("[Web evidence]\n" + "\n\n".join(sources))

    return "\n\n".join(parts).strip()


def _clean_model_answer(text: str) -> str:
    text = str(text or "").strip()

    # Remove common answer headings, but keep the answer itself.
    text = re.sub(
        r"(?im)^\s*(?:#{1,3}\s*)?(?:answer|response)\s*:?\s*",
        "",
        text,
    )

    # Remove a trailing sources section.
    text = re.split(
        r"(?im)^\s*(?:#{1,3}\s*)?(?:sources|references|web sources)\s*:?\s*$",
        text,
        maxsplit=1,
    )[0]

    lines = []
    for line in text.splitlines():
        line = re.sub(r"https?://\S+", "", line).strip()
        if line:
            lines.append(line)

    text = "\n".join(lines).strip()

    # Remove duplicate paragraphs.
    unique = []
    seen = set()
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        normalized = re.sub(r"\W+", " ", paragraph.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(paragraph)

    return "\n\n".join(unique).strip()


def _fallback_answer(
    question: str,
    web_results: list[dict[str, str]],
    docs: list[dict],
) -> str:
    if docs:
        content = _clean_evidence(str(docs[0].get("text", "")))
        if content:
            return "Based on your uploaded document:\n\n" + content[:1200]
        return "I couldn't extract readable text from your uploaded document."

    if web_results:
        try:
            result = extractive_answer(question, web_results)
            if result and result.strip():
                return _clean_model_answer(result)
        except Exception:
            logger.exception("Extractive fallback failed")

    return (
        "The local AI could not generate an answer. "
        "Please check the Flask terminal for the error and try again."
    )


def answer(
    question: str,
    conversation_id: int | None = None,
    use_web: bool = True,
    use_model: bool = True,
    document_ids: list[int] | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    web_results: list[dict[str, str]] = []
    search_error: str | None = None

    if use_web:
        try:
            web_results, _engine = web_search(
                question, max_results=MAX_WEB_RESULTS
            )
        except SearchError as exc:
            search_error = str(exc)
            logger.warning("Web search failed: %s", exc)
        except Exception as exc:
            search_error = str(exc)
            logger.exception("Unexpected web search failure")

    docs: list[dict] = []
    if conversation_id:
        try:
            all_docs = database.latest_documents(conversation_id, limit=20)
            if document_ids:
                requested_ids = set(document_ids)
                docs = [
                    doc for doc in all_docs
                    if doc.get("id") in requested_ids
                ]
            elif all_docs:
                docs = all_docs[:1]
        except Exception:
            logger.exception("Could not load uploaded documents")

    context = _build_context(question, conversation_id, web_results, docs)

    answer_text: str | None = None
    model_name: str | None = None
    mode = "search"

    if use_model:
        try:
            model_name = model_available()
            logger.info("Askwise selected local model: %r", model_name)

            if model_name:
                user_prompt = (
                    f"CURRENT QUESTION:\n{question}\n\n"
                    f"REFERENCE MATERIAL:\n"
                    f"{context if context else 'No reference material available.'}\n\n"
                    "Answer the CURRENT QUESTION only. "
                    "For general knowledge and simple maths, use your own "
                    "knowledge when appropriate. Use reference material only "
                    "when relevant. Do not list sources or URLs."
                )

                raw = ollama_chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model_name,
                )

                logger.info("Askwise raw model response: %r", raw[:500])
                cleaned = _clean_model_answer(raw)
                logger.info("Askwise cleaned model response: %r", cleaned[:500])

                if cleaned:
                    answer_text = cleaned
                    mode = "ai" if web_results else "local"
                else:
                    logger.warning("Ollama returned an empty cleaned answer.")
            else:
                logger.warning(
                    "No installed preferred Ollama model was detected."
                )

        except Exception:
            logger.exception("Askwise local model generation failed")

    if answer_text is None:
        answer_text = _fallback_answer(question, web_results, docs)
        mode = "search"

    return {
        "answer": answer_text,
        "sources": web_results,
        "mode": mode,
        "model": model_name,
        "search_error": search_error,
        "conversation_id": conversation_id,
    }


def run_and_store(
    question: str,
    conversation_id: int,
    use_web: bool = True,
    use_model: bool = True,
    document_ids: list[int] | None = None,
) -> dict[str, Any]:
    database.add_message(conversation_id, "user", question)

    result = answer(
        question,
        conversation_id=conversation_id,
        use_web=use_web,
        use_model=use_model,
        document_ids=document_ids,
    )

    database.add_message(
        conversation_id,
        "assistant",
        result["answer"],
        mode=result["mode"],
        sources=result.get("sources"),
    )

    return result
