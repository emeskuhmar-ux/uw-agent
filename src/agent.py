"""Main agent. Routes between corpus retrieval and web search, then synthesizes."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator

from src import config, llm
from src.corpus import Corpus, Chunk
from src.ingest import SourceRecord, ingest_batch
from .search import openalex, arxiv_search, semantic_scholar, crossref, unpaywall, tavily_search


@dataclass
class AgentTrace:
    """Diagnostic info for the UI sidebar."""
    on_topic: bool = True
    corpus_hits: int = 0
    corpus_confidence: float = 0.0
    used_web: bool = False
    web_results_found: int = 0
    pdfs_ingested: int = 0
    sources_used: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Agent:
    def __init__(self):
        self.corpus = Corpus()

    # ---------- Web search aggregation ----------

    def _web_search_all(self, query: str) -> list[SourceRecord]:
        """Hit all configured search backends in parallel-ish fashion."""
        all_sources: dict[str, SourceRecord] = {}

        # OpenAlex
        for w in openalex.search(query, max_results=config.MAX_OPENALEX_RESULTS):
            if not w.title:
                continue
            pdf_url = w.pdf_url
            # If no OA PDF but we have a DOI, ask Unpaywall
            if not pdf_url and w.doi:
                pdf_url = unpaywall.find_oa_pdf(w.doi)
            all_sources[w.id] = SourceRecord(
                source_id=w.id, source_type="openalex",
                title=w.title, authors=w.authors, year=w.year,
                url=w.url, pdf_url=pdf_url, abstract=w.abstract, venue=w.venue,
            )

        # arXiv
        for p in arxiv_search.search(query, max_results=config.MAX_ARXIV_RESULTS):
            if p.id not in all_sources:
                all_sources[p.id] = SourceRecord(
                    source_id=p.id, source_type="arxiv",
                    title=p.title, authors=p.authors, year=p.year,
                    url=p.url, pdf_url=p.pdf_url, abstract=p.abstract, venue="arXiv",
                )

        # Semantic Scholar
        for p in semantic_scholar.search(query, max_results=config.MAX_S2_RESULTS):
            if p.id not in all_sources:
                all_sources[p.id] = SourceRecord(
                    source_id=p.id, source_type="s2",
                    title=p.title, authors=p.authors, year=p.year,
                    url=p.url, pdf_url=p.pdf_url, abstract=p.abstract, venue=p.venue,
                )

        # Crossref — typically abstract-only, useful as backup
        for w in crossref.search(query, max_results=4):
            if w.id not in all_sources and w.abstract:
                pdf_url = unpaywall.find_oa_pdf(w.doi) if w.doi else None
                all_sources[w.id] = SourceRecord(
                    source_id=w.id, source_type="crossref",
                    title=w.title, authors=w.authors, year=w.year,
                    url=w.url, pdf_url=pdf_url, abstract=w.abstract, venue=w.venue,
                )

        # Tavily for grey literature
        for h in tavily_search.search(query, max_results=config.MAX_TAVILY_RESULTS):
            sid = f"web:{hash(h.url) & 0xffffffff:x}"
            if sid not in all_sources:
                all_sources[sid] = SourceRecord(
                    source_id=sid, source_type="tavily",
                    title=h.title, authors="", year=None,
                    url=h.url, pdf_url=None, abstract=h.snippet, venue="Web",
                )

        return list(all_sources.values())

    # ---------- Prompt building ----------

    def _build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        """Pack retrieved chunks into a citation-aware prompt."""
        if not chunks:
            return f"Question: {question}\n\nNo relevant sources were found. Tell the user honestly."
        parts = ["You will answer based ONLY on the following numbered sources.\n"]
        for i, c in enumerate(chunks, 1):
            year = f", {c.year}" if c.year else ""
            authors = c.authors[:80] + ("..." if len(c.authors) > 80 else "")
            parts.append(
                f"[{i}] {c.source_title} ({authors}{year}) — {c.source_url}\n"
                f"{c.text.strip()[:1500]}\n"
            )
        parts.append(f"\nQuestion: {question}\n")
        parts.append(
            "Instructions:\n"
            "- Cite using [n] inline.\n"
            "- If sources don't cover the question well, say so.\n"
            "- Be concise, technical, and include units in any numerical results.\n"
        )
        return "\n".join(parts)

    # ---------- Main entry point ----------

    def answer(self, question: str) -> Iterator[tuple[str, AgentTrace]]:
        """Yield (text_chunk, trace) tuples. The trace updates as we go."""
        trace = AgentTrace()

        # 1. Topic gate
        if not llm.is_on_topic(question):
            trace.on_topic = False
            trace.notes.append("Question classified as off-topic for underwater systems.")
            yield (
                "I'm focused on underwater systems engineering (AUVs, ROVs, "
                "underwater drones, marine robotics, hydrodynamics, etc.). "
                "Could you ask something in that domain?",
                trace,
            )
            return

        # 2. Try local corpus first
        corpus_hits = self.corpus.hybrid_search(question)
        trace.corpus_hits = len(corpus_hits)
        trace.corpus_confidence = self.corpus.confidence(corpus_hits)
        trace.notes.append(
            f"Corpus: {trace.corpus_hits} chunks, confidence={trace.corpus_confidence:.2f}"
        )

        chunks_for_llm: list[Chunk] = []
        if (
            trace.corpus_confidence >= config.CORPUS_CONFIDENCE_THRESHOLD
            and trace.corpus_hits >= config.MIN_CORPUS_HITS
        ):
            chunks_for_llm = corpus_hits
            trace.notes.append("Using corpus only (confidence high).")
        else:
            # 3. Web search
            trace.used_web = True
            trace.notes.append("Confidence low → web search.")
            web_sources = self._web_search_all(question)
            trace.web_results_found = len(web_sources)
            trace.notes.append(f"Found {len(web_sources)} web sources.")

            if web_sources:
                # Prefer sources with OA PDFs first, then abstract-only
                with_pdf = [s for s in web_sources if s.pdf_url]
                without_pdf = [s for s in web_sources if not s.pdf_url]
                ordered = with_pdf[: config.MAX_PDF_DOWNLOADS_PER_QUERY] + without_pdf
                n_added = ingest_batch(self.corpus, ordered, max_pdfs=config.MAX_PDF_DOWNLOADS_PER_QUERY)
                trace.pdfs_ingested = sum(1 for s in ordered[: config.MAX_PDF_DOWNLOADS_PER_QUERY] if s.pdf_url)
                trace.notes.append(f"Ingested {n_added} new chunks from web.")
                # Re-run corpus search now that we have new content
                chunks_for_llm = self.corpus.hybrid_search(question)
            else:
                chunks_for_llm = corpus_hits  # fall back to whatever we had

        # 4. Record sources for the UI
        seen_sources = {}
        for i, c in enumerate(chunks_for_llm, 1):
            if c.source_id not in seen_sources:
                seen_sources[c.source_id] = {
                    "index": i,
                    "title": c.source_title,
                    "authors": c.authors,
                    "year": c.year,
                    "url": c.source_url,
                    "type": c.source_type,
                }
        trace.sources_used = list(seen_sources.values())

        # 5. Synthesize with the LLM
        prompt = self._build_prompt(question, chunks_for_llm)
        full_answer = ""
        for piece in llm.stream(prompt):
            full_answer += piece
            yield (piece, trace)
