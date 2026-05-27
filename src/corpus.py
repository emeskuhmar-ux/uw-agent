"""Local corpus with hybrid (vector + BM25) retrieval."""
from __future__ import annotations
import pickle
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_id: str
    source_title: str
    source_url: str
    source_type: str
    year: int | None = None
    authors: str = ""
    score: float = 0.0


_embedder: SentenceTransformer | None = None


def _embed_model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    return _embed_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


class Corpus:
    def __init__(self):
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection = config.QDRANT_COLLECTION
        self._ensure_collection()
        self.bm25: BM25Okapi | None = None
        self.bm25_chunks: list[Chunk] = []
        self._load_bm25()

    def _ensure_collection(self):
        collections = {c.name for c in self.client.get_collections().collections}
        if self.collection not in collections:
            dim = _embed_model().get_sentence_embedding_dimension()
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _load_bm25(self):
        if config.BM25_INDEX_PATH.exists():
            try:
                with open(config.BM25_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.bm25_chunks = data["chunks"]
            except Exception:
                self.bm25 = None
                self.bm25_chunks = []

    def _save_bm25(self):
        if not self.bm25_chunks:
            return
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.bm25_chunks}, f)

    def _rebuild_bm25(self):
        all_chunks = self._all_chunks()
        if not all_chunks:
            self.bm25 = None
            self.bm25_chunks = []
            return
        tokenized = [c.text.lower().split() for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_chunks = all_chunks
        self._save_bm25()

    def _all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in batch:
                p = point.payload or {}
                chunks.append(Chunk(
                    chunk_id=str(point.id),
                    text=p.get("text", ""),
                    source_id=p.get("source_id", ""),
                    source_title=p.get("source_title", ""),
                    source_url=p.get("source_url", ""),
                    source_type=p.get("source_type", "manual"),
                    year=p.get("year"),
                    authors=p.get("authors", ""),
                ))
            if offset is None:
                break
        return chunks

    def add_chunks(self, chunks: list[Chunk], rebuild_bm25: bool = True):
        if not chunks:
            return
        vectors = _embed([c.text for c in chunks])
        points = []
        for c, v in zip(chunks, vectors):
            points.append(qmodels.PointStruct(
                id=c.chunk_id,
                vector=v.tolist(),
                payload={
                    "text": c.text,
                    "source_id": c.source_id,
                    "source_title": c.source_title,
                    "source_url": c.source_url,
                    "source_type": c.source_type,
                    "year": c.year,
                    "authors": c.authors,
                },
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        if rebuild_bm25:
            self._rebuild_bm25()

    def has_source(self, source_id: str) -> bool:
        hits, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="source_id",
                    match=qmodels.MatchValue(value=source_id),
                )]
            ),
            limit=1,
        )
        return len(hits) > 0

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    def count_sources(self) -> int:
        seen = {c.source_id for c in self._all_chunks()}
        return len(seen)

    def vector_search(self, query: str, k: int) -> list[Chunk]:
        qv = _embed([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=qv.tolist(),
            limit=k,
            with_payload=True,
        )
        hits = response.points
        results = []
        for h in hits:
            p = h.payload or {}
            results.append(Chunk(
                chunk_id=str(h.id),
                text=p.get("text", ""),
                source_id=p.get("source_id", ""),
                source_title=p.get("source_title", ""),
                source_url=p.get("source_url", ""),
                source_type=p.get("source_type", "manual"),
                year=p.get("year"),
                authors=p.get("authors", ""),
                score=float(h.score),
            ))
        return results

    def bm25_search(self, query: str, k: int) -> list[Chunk]:
        if not self.bm25 or not self.bm25_chunks:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        if scores.max() == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        max_score = scores.max()
        for i in top_idx:
            if scores[i] <= 0:
                continue
            c = self.bm25_chunks[i]
            c2 = Chunk(**{**asdict(c), "score": float(scores[i] / max_score)})
            results.append(c2)
        return results

    def hybrid_search(self, query: str) -> list[Chunk]:
        vec = self.vector_search(query, config.TOP_K_VECTOR)
        bm = self.bm25_search(query, config.TOP_K_BM25)

        rrf: dict[str, dict[str, Any]] = {}
        k_rrf = 60

        for rank, c in enumerate(vec):
            rrf.setdefault(c.chunk_id, {"chunk": c, "score": 0.0})
            rrf[c.chunk_id]["score"] += 1.0 / (k_rrf + rank)
        for rank, c in enumerate(bm):
            rrf.setdefault(c.chunk_id, {"chunk": c, "score": 0.0})
            rrf[c.chunk_id]["score"] += 1.0 / (k_rrf + rank)

        ranked = sorted(rrf.values(), key=lambda x: x["score"], reverse=True)
        results = []
        for r in ranked[: config.TOP_K_FINAL]:
            c = r["chunk"]
            c.score = r["score"]
            results.append(c)
        return results

    def confidence(self, hits: list[Chunk]) -> float:
        if len(hits) < config.MIN_CORPUS_HITS:
            return 0.0
        scores = []
        for h in hits[:3]:
            scores.append(min(h.score * 60, 1.0))
        return float(np.mean(scores)) if scores else 0.0
