"""OpenAlex: free, open metadata for ~250M scholarly works. No key required."""
from __future__ import annotations
import httpx
from dataclasses import dataclass

from .. import config


@dataclass
class Work:
    id: str            # e.g. "openalex:W2741809807"
    title: str
    abstract: str
    authors: str
    year: int | None
    url: str           # landing page URL
    pdf_url: str | None  # open-access PDF if available
    doi: str | None
    venue: str
    is_oa: bool


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys()))


def search(query: str, max_results: int = 8) -> list[Work]:
    """Search OpenAlex for scholarly works."""
    params = {
        "search": query,
        "per-page": max_results,
        "filter": "type:article|review|book-chapter",
        "mailto": config.CONTACT_EMAIL,
    }
    url = "https://api.openalex.org/works"
    try:
        r = httpx.get(url, params=params, timeout=20.0)
        r.raise_for_status()
    except Exception as e:
        print(f"[openalex] error: {e}")
        return []

    works = []
    for w in r.json().get("results", []):
        oa = w.get("open_access", {}) or {}
        pdf_url = oa.get("oa_url")
        authors_list = [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])[:5]]
        works.append(Work(
            id=f"openalex:{w.get('id', '').split('/')[-1]}",
            title=w.get("title") or "",
            abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
            authors=", ".join(filter(None, authors_list)),
            year=w.get("publication_year"),
            url=w.get("doi") or w.get("id") or "",
            pdf_url=pdf_url,
            doi=w.get("doi"),
            venue=(w.get("primary_location") or {}).get("source", {}).get("display_name", "") or "",
            is_oa=bool(oa.get("is_oa")),
        ))
    return works
