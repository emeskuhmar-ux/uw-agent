"""Crossref - free DOI metadata. Useful for filling gaps from other sources."""
from __future__ import annotations
import httpx
from dataclasses import dataclass

from .. import config


@dataclass
class CrossrefWork:
    id: str
    title: str
    abstract: str
    authors: str
    year: int | None
    url: str
    doi: str
    venue: str


def search(query: str, max_results: int = 6) -> list[CrossrefWork]:
    params = {
        "query": query,
        "rows": max_results,
        "select": "DOI,title,abstract,author,issued,container-title,URL",
    }
    headers = {"User-Agent": f"underwater-agent ({config.CONTACT_EMAIL})"}
    try:
        r = httpx.get("https://api.crossref.org/works", params=params, headers=headers, timeout=20.0)
        r.raise_for_status()
    except Exception as e:
        print(f"[crossref] error: {e}")
        return []

    items = r.json().get("message", {}).get("items", [])
    out = []
    for it in items:
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (it.get("author") or [])[:5]
        )
        year = None
        issued = it.get("issued", {}).get("date-parts", [[None]])
        if issued and issued[0] and issued[0][0]:
            year = issued[0][0]
        out.append(CrossrefWork(
            id=f"doi:{it.get('DOI', '')}",
            title=(it.get("title") or [""])[0],
            abstract=(it.get("abstract") or "").replace("<jats:p>", "").replace("</jats:p>", ""),
            authors=authors,
            year=year,
            url=it.get("URL", ""),
            doi=it.get("DOI", ""),
            venue=(it.get("container-title") or [""])[0],
        ))
    return out
