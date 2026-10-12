
"""Free web search with relevance filtering and automatic fallback."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

import requests
from ddgs import DDGS

UA = {"User-Agent": "Askwise/1.0 (educational search client)"}


class SearchError(Exception):
    """Raised when web search cannot provide usable results."""


def _normalize(
    raw: list[dict[str, Any]], engine: str
) -> list[dict[str, str]]:
    results = []
    seen_urls = set()

    for item in raw or []:
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(
            item.get("body") or item.get("snippet") or item.get("extract") or ""
        ).strip()

        if not title or not snippet:
            continue
        if urlsplit(url).scheme.lower() not in {"http", "https"}:
            continue

        key = url.rstrip("/").lower()
        if key in seen_urls:
            continue

        seen_urls.add(key)
        results.append({
            "title": title,
            "url": url,
            "snippet": snippet[:1200],
            "engine": engine,
        })

    return results


def _duckduckgo(question: str, max_results: int) -> list[dict[str, str]]:
    with DDGS() as search:
        raw = search.text(question, max_results=max_results)
    return _normalize(raw, "duckduckgo")


def _wikipedia(question: str, max_results: int) -> list[dict[str, str]]:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": question,
            "format": "json",
            "srlimit": max_results,
        },
        headers=UA,
        timeout=12,
    )
    response.raise_for_status()

    hits = response.json().get("query", {}).get("search", [])
    results = []

    for hit in hits:
        page_id = hit.get("pageid")
        title = str(hit.get("title") or "").strip()
        if not page_id or not title:
            continue

        # Avoid a programming-language question returning unrelated
        # entertainment pages simply because they share a word.
        if _is_programming_question(question):
            if title.lower() in {
                "monty python",
                "pythonidae",
                "reticulated python",
                "python (genus)",
            }:
                continue

        try:
            detail = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exintro": 1,
                    "pageids": page_id,
                    "format": "json",
                },
                headers=UA,
                timeout=12,
            )
            detail.raise_for_status()

            pages = detail.json().get("query", {}).get("pages", {})
            extract = pages.get(str(page_id), {}).get("extract", "").strip()

            if extract:
                results.append({
                    "title": title,
                    "url": f"https://en.wikipedia.org/?curid={page_id}",
                    "snippet": extract[:1200],
                    "engine": "wikipedia",
                })
        except requests.RequestException:
            continue

    return results


def _is_programming_question(question: str) -> bool:
    q = question.lower()
    programming_clues = (
        "programming", "programming language", "coding", "code",
        "developer", "software", "syntax", "python language",
        "python mein", "python me", "python kya hai",
        "what is python", "learn python", "python tutorial",
        "python popular", "python used for",
    )
    return any(clue in q for clue in programming_clues)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#.]+", text.lower()))


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\[[^\]]{1,30}\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _relevance(question: str, result: dict[str, str]) -> float:
    q = question.lower()
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    combined = f"{title} {snippet}"
    q_terms = _terms(question)

    # Ignore common question words when scoring keyword matches.
    stop_words = {
        "what", "which", "when", "where", "why", "who", "how",
        "does", "do", "did", "is", "are", "was", "were", "the",
        "and", "for", "from", "with", "that", "this", "you", "your",
        "about", "tell", "me", "explain", "give", "some",
    }
    meaningful = {w for w in q_terms if w not in stop_words}
    if not meaningful:
        meaningful = q_terms

    score = 0.0
    for term in meaningful:
        if term in title:
            score += 4.0
        elif term in snippet:
            score += 1.0

    # Prefer exact phrases and titles that match the question's topic.
    if q.strip(" ?.!") in title:
        score += 5.0

    if _is_programming_question(question):
        if "programming language" in combined:
            score += 8.0
        if "software" in combined or "computer" in combined:
            score += 2.0
        if any(word in title for word in (
            "monty python", "pythonidae", "reticulated python",
            "python (genus)",
        )):
            score -= 15.0

    # Normalize by query size so long questions do not dominate.
    return score / max(len(meaningful), 1)


def _sentence_similarity(first: str, second: str) -> float:
    a = _terms(first)
    b = _terms(second)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def web_search(
    question: str, max_results: int = 6
) -> tuple[list[dict[str, str]], str]:
    question = question.strip()
    if not question:
        raise SearchError("Please enter a question.")

    errors = []

    for engine_name, engine_function in (
        ("duckduckgo", _duckduckgo),
        ("wikipedia", _wikipedia),
    ):
        try:
            results = engine_function(question, max_results)
            if results:
                results.sort(
                    key=lambda item: _relevance(question, item),
                    reverse=True,
                )

                # For clearly irrelevant results, don't pretend they are useful.
                best_score = _relevance(question, results[0])
                if best_score > 0:
                    return results, engine_name

                errors.append(f"{engine_name}: results were not relevant")
            else:
                errors.append(f"{engine_name}: no results")
        except Exception as exc:
            errors.append(f"{engine_name}: {exc}")

    raise SearchError(
        "No relevant web results were found. Try rephrasing your question. "
        + " | ".join(errors)
    )


def extractive_answer(
    question: str, results: list[dict[str, str]]
) -> str:
    """Return a short answer from relevant, non-duplicate snippets."""
    if not results:
        return "I couldn't find useful information. Try rephrasing your question."

    ranked = sorted(
        results,
        key=lambda item: _relevance(question, item),
        reverse=True,
    )

    answer_sentences = []
    programming_question = _is_programming_question(question)

    for result in ranked:
        title = result.get("title", "")
        snippet = _clean_text(result.get("snippet", ""))

        if programming_question and any(
            word in title.lower()
            for word in (
                "monty python", "pythonidae", "reticulated python",
                "python (genus)",
            )
        ):
            continue

        if not snippet or _relevance(question, result) <= 0:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", snippet)

        for sentence in sentences:
            sentence = sentence.strip(" \t\n-•")
            if len(sentence) < 30:
                continue

            # Avoid weak sentences that do not mention the topic.
            if programming_question:
                lower = sentence.lower()
                if not any(word in lower for word in (
                    "python", "programming", "language", "software",
                    "code", "developer", "computer",
                )):
                    continue

            if any(
                _sentence_similarity(sentence, previous) >= 0.65
                for previous in answer_sentences
            ):
                continue

            answer_sentences.append(sentence)
            if len(answer_sentences) >= 4:
                break

        if len(answer_sentences) >= 4:
            break

    if not answer_sentences:
        return (
            "I couldn't find enough relevant information to answer that "
            "reliably. Try adding more detail to your question."
        )

    return " ".join(answer_sentences)
