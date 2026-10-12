"""Level 2 — local AI inference via Ollama (free, open-weight models).

Ollama runs locally with no API key and no cloud dependency.
This module auto-detects:
  * whether the Ollama server is reachable at OLLAMA_BASE_URL (default
    http://127.0.0.1:11434)
  * whether the preferred model has been pulled
  * the best free model for the machine's RAM (used by the setup script)

All model usage here is free/open-weight (e.g. Llama, Qwen, Phi, Gemma).
"""
from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - only used for read-only `ollama` CLI probes with a fixed arg list
import threading
from typing import Any

import requests

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

# Free, open-weight models ONLY, smallest-first by RAM need.
PREFERRED_MODELS: list[str] = [
    m.strip() for m in os.getenv(
        "ASKWISE_MODELS",
        "qwen3:0.6b,llama3.2:3b,qwen2.5:3b,phi3:3.8b,gemma2:2b,llama3.2:1b",
    ).split(",")
    if m.strip()
]

_gen_lock = threading.Lock()


def _try_request(method: str, path: str, **kwargs: Any) -> requests.Response | None:
    timeout = kwargs.pop("timeout", 3)
    try:
        return requests.request(method, f"{BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException:
        return None


def ollama_reachable() -> bool:
    resp = _try_request("GET", "/api/version", timeout=2)
    return resp is not None and resp.ok


def installed_models() -> list[str]:
    resp = _try_request("GET", "/api/tags")
    if resp is None or not resp.ok:
        return []
    try:
        tags = resp.json().get("models", []) or []
    except ValueError:
        return []
    return [m.get("name", "") for m in tags if m.get("name")]


def pick_model() -> str | None:
    """First preferred model that is actually installed locally."""
    installed = installed_models()
    for wanted in PREFERRED_MODELS:
        for have in installed:
            if have.split(":")[0] == wanted.split(":")[0] and (
                have.split(":")[1:2] == wanted.split(":")[1:2] or ":" not in wanted
            ):
                return have
    return None


def model_available() -> str | None:
    """Return the usable model name if Ollama + model are ready, else None."""
    if not ollama_reachable():
        return None
    return pick_model()


def chat(messages: list[dict[str, str]], model: str | None = None,
         temperature: float = 0.2, num_ctx: int = 4096) -> str:
    """Blocking chat completion against the local Ollama server.

    Raises RuntimeError when Ollama or the model is not available.
    """
    model = model or model_available()
    if not model:
        raise RuntimeError("Ollama is not running or no model is installed. Run `ollama pull llama3.2:3b`.")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    resp = _try_request("POST", "/api/chat", json=payload) or requests.post(
        f"{BASE_URL}/api/chat", json=payload, timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("message") or {}).get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Unexpected response from Ollama.")
    return content.strip()


def best_model_for_ram() -> str:
    """Recommend a free open-weight model based on system RAM (Windows-friendly)."""
    total_gb = 0.0
    try:
        import psutil  # optional; use if available

        total_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        try:
            # Windows fallback without psutil
            out = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, check=False, timeout=10,
            ).stdout
            for token in out.split():
                if token.isdigit():
                    total_gb = int(token) / (1024 ** 3)
                    break
        except Exception:
            pass

    if total_gb >= 30:
        return "llama3.1:8b"
    if total_gb >= 14:
        return "llama3.2:3b"
    return "llama3.2:1b"


def recommend_winget() -> str:
    model = best_model_for_ram()
    return (
        f"winget install --id Ollama.Ollama --exact --source winget\n"
        f"ollama pull {model}"
    )


def strip_citations(text: str) -> str:
    """Remove citation markers like [1] if any slipped in from the search prompt."""
    return re.sub(r"\s*\[\d+\]", "", text)
