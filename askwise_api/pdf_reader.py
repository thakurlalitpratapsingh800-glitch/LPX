"""Level 1 — PDF document reading, fully free software.

Preferred library: pypdf (pure Python, BSD license, no paid services).
If pypdf is not installed it returns a helpful error instead of crashing.
"""
from __future__ import annotations

import io

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB per PDF keeps things simple and safe
MAX_CHARS = 120_000


def extract_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF is larger than 10 MB. Split it and upload the parts you need.")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF support needs the free 'pypdf' package: pip install pypdf") from exc

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError("That file doesn't look like a valid PDF.") from exc

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    if not pages:
        raise ValueError(
            "No selectable text found in this PDF. Scanned/image-only PDFs need OCR, "
            "which is a limitation of free text extraction here."
        )
    return "\n\n".join(pages)[:MAX_CHARS]
