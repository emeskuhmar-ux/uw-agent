"""arXiv search - free, no key. Always returns PDF URLs."""
from __future__ import annotations
from dataclasses import dataclass
import arxiv


@dataclass
class ArxivPaper:
    id: str            # "arxiv:2401.12345"
    title: str
    abstract: str
    authors: str
    year: int | None
    url: str
    pdf_url: str
    venue: str


def search(query: str, max_results: int = 6) -> list[ArxivPaper]:
    """Search arXiv for papers."""
    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=2)
        s = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        for r in client.results(s):
            short_id = r.entry_id.split("/")[-1].split("v")[0]
            papers.append(ArxivPaper(
                id=f"arxiv:{short_id}",
                title=r.title or "",
                abstract=(r.summary or "").replace("\n", " "),
                authors=", ".join(a.name for a in (r.authors or [])[:5]),
                year=r.published.year if r.published else None,
                url=r.entry_id,
                pdf_url=r.pdf_url,
                venue="arXiv",
            ))
        return papers
    except Exception as e:
        print(f"[arxiv] error: {e}")
        return []
