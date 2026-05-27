"""Download PDFs from URLs and parse them to clean markdown.

Uses Docling for robust parsing of equations, tables, and multi-column layouts.
Falls back to pypdf if Docling fails on a particular file.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import httpx

from .. import config


def _safe_filename(source_id: str) -> str:
    safe = source_id.replace(":", "_").replace("/", "_")
    return f"{safe}.pdf"


def download_pdf(url: str, source_id: str) -> Path | None:
    """Download a PDF if not already cached. Returns path or None."""
    if not url:
        return None
    fname = _safe_filename(source_id)
    path = config.CORPUS_DIR / fname
    if path.exists() and path.stat().st_size > 1000:
        return path
    try:
        headers = {"User-Agent": f"underwater-agent ({config.CONTACT_EMAIL})"}
        with httpx.stream("GET", url, headers=headers, timeout=30.0, follow_redirects=True) as r:
            if r.status_code != 200:
                return None
            ctype = r.headers.get("content-type", "").lower()
            # Sometimes URLs return HTML landing pages — skip those
            if "pdf" not in ctype and "octet-stream" not in ctype:
                return None
            with open(path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        if path.stat().st_size < 1000:
            path.unlink(missing_ok=True)
            return None
        return path
    except Exception as e:
        print(f"[pdf-fetch] error {url}: {e}")
        return None


def parse_pdf(path: Path) -> str:
    """Parse PDF to markdown text. Tries Docling, falls back to pypdf."""
    # Try Docling first
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        md = result.document.export_to_markdown()
        if md and len(md) > 200:
            return md
    except Exception as e:
        print(f"[docling] failed on {path.name}: {e}, falling back to pypdf")

    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        print(f"[pypdf] also failed on {path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Word-based chunking. Simple but effective for technical content."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 100:  # skip tiny chunks
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks
