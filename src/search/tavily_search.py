"""Tavily web search for grey literature, technical blogs, GitHub, etc.

Free tier: 1000 searches/month. Falls back gracefully if no key.
"""
from __future__ import annotations
from dataclasses import dataclass

from .. import config


@dataclass
class WebHit:
    title: str
    url: str
    snippet: str
    score: float


def search(query: str, max_results: int = 5) -> list[WebHit]:
    if not config.TAVILY_API_KEY:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        # Use 'advanced' for deeper search of engineering content
        result = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
        hits = []
        for r in result.get("results", []):
            hits.append(WebHit(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            ))
        return hits
    except Exception as e:
        print(f"[tavily] error: {e}")
        return []
