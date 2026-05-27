"""Ingestion: turn search results into Corpus chunks.

Two ingestion modes:
 1. From a search result with a PDF URL → download + parse + chunk + index
 2. From abstract-only metadata → index just the abstract as a single chunk
"""
from __future__ import annotations
import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from src.corpus import Corpus, Chunk
from .utils.pdf_fetch import download_pdf, parse_pdf, chunk_text


@dataclass
class SourceRecord:
    """Unified record across all search backends."""
    source_id: str
    source_type: str
    title: str
    authors: str
    year: int | None
    url: str
    pdf_url: str | None
    abstract: str
    venue: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "abstract_len": len(self.abstract),
            "venue": self.venue,
        }


def ingest_source(corpus: Corpus, src: SourceRecord, try_pdf: bool = True) -> int:
    """Ingest a single source. Returns number of chunks added."""
    if corpus.has_source(src.source_id):
        return 0

    chunks_to_add: list[Chunk] = []
    full_text = ""

    if try_pdf and src.pdf_url:
        path = download_pdf(src.pdf_url, src.source_id)
        if path:
            full_text = parse_pdf(path)

    if full_text and len(full_text) > 500:
        pieces = chunk_text(full_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for piece in pieces:
            chunks_to_add.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=piece,
                source_id=src.source_id,
                source_title=src.title,
                source_url=src.url,
                source_type=src.source_type,
                year=src.year,
                authors=src.authors,
            ))
    elif src.abstract:
        # Index abstract as a single chunk so we at least have something
        chunks_to_add.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=f"{src.title}\n\n{src.abstract}",
            source_id=src.source_id,
            source_title=src.title,
            source_url=src.url,
            source_type=src.source_type,
            year=src.year,
            authors=src.authors,
        ))

    if chunks_to_add:
        corpus.add_chunks(chunks_to_add, rebuild_bm25=False)
        _log_ingest(src, len(chunks_to_add))
    return len(chunks_to_add)


def ingest_batch(corpus: Corpus, sources: list[SourceRecord], max_pdfs: int | None = None) -> int:
    """Ingest a batch. Limits how many full PDFs to fetch."""
    total = 0
    pdfs_fetched = 0
    for src in sources:
        try_pdf = src.pdf_url and (max_pdfs is None or pdfs_fetched < max_pdfs)
        n = ingest_source(corpus, src, try_pdf=bool(try_pdf))
        if n > 1:
            pdfs_fetched += 1
        total += n
    if total > 0:
        corpus._rebuild_bm25()
    return total


def _log_ingest(src: SourceRecord, n_chunks: int):
    config.INGESTED_LOG.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if config.INGESTED_LOG.exists():
        try:
            entries = json.loads(config.INGESTED_LOG.read_text())
        except Exception:
            entries = []
    entries.append({**src.to_dict(), "n_chunks": n_chunks})
    config.INGESTED_LOG.write_text(json.dumps(entries, indent=2))


def init_empty():
    """Create an empty corpus (just initializes Qdrant)."""
    corpus = Corpus()
    print(f"Initialized corpus at {config.QDRANT_PATH}")
    print(f"Current chunks: {corpus.count()}")
    print(f"Current unique sources: {corpus.count_sources()}")


def ingest_local_pdfs():
    """Ingest any PDFs the user dropped manually into ./data/"""
    corpus = Corpus()
    pdfs = list(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {config.DATA_DIR}")
        return
    sources = []
    for p in pdfs:
        sources.append(SourceRecord(
            source_id=f"manual:{p.stem}",
            source_type="manual",
            title=p.stem.replace("_", " "),
            authors="",
            year=None,
            url=str(p),
            pdf_url=str(p),
            abstract="",
        ))
    n = ingest_batch(corpus, sources, max_pdfs=len(sources))
    print(f"Ingested {n} chunks from {len(pdfs)} local PDFs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Initialize empty corpus")
    parser.add_argument("--local", action="store_true", help="Ingest PDFs in ./data/")
    args = parser.parse_args()
    if args.init:
        init_empty()
    elif args.local:
        ingest_local_pdfs()
    else:
        init_empty()
