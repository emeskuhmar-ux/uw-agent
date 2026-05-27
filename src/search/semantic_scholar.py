"""Semantic Scholar - free; key recommended for higher rate limits."""
from __future__ import annotations
import httpx
from dataclasses import dataclass

from .. import config


@dataclass
class S2Paper:
    id: str            # "s2:abc123"
    title: str
    abstract: str
    authors: str
    year: int | None
    url: str
    pdf_url: str | None
    venue: str
    is_oa: bool


def search(query: str, max_results: int = 6) -> list[S2Paper]:
    """Search Semantic Scholar."""
    headers = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,url,venue,openAccessPdf,isOpenAccess,paperId",
    }
    try:
        r = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, headers=headers, timeout=20.0,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[s2] error: {e}")
        return []

    papers = []
    for p in r.json().get("data", []):
        oa_pdf = (p.get("openAccessPdf") or {}).get("url")
        authors_list = [a.get("name", "") for a in (p.get("authors") or [])[:5]]
        papers.append(S2Paper(
            id=f"s2:{p.get('paperId', '')}",
            title=p.get("title") or "",
            abstract=p.get("abstract") or "",
            authors=", ".join(filter(None, authors_list)),
            year=p.get("year"),
            url=p.get("url") or "",
            pdf_url=oa_pdf,
            venue=p.get("venue") or "",
            is_oa=bool(p.get("isOpenAccess")),
        ))
    return papers
