"""Central configuration. All paths and settings live here."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
CORPUS_DIR = ROOT_DIR / "corpus"
STORAGE_DIR = ROOT_DIR / "storage"
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"

for d in (CORPUS_DIR, STORAGE_DIR, LOGS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

QDRANT_PATH = str(STORAGE_DIR / "qdrant")
QDRANT_COLLECTION = "underwater_docs"
BM25_INDEX_PATH = STORAGE_DIR / "bm25_index.pkl"
INGESTED_LOG = LOGS_DIR / "ingested.json"

# --- API keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "anonymous@example.com").strip()

# --- Models ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()

# --- Retrieval settings ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 6           # how many chunks fed to the LLM
CORPUS_CONFIDENCE_THRESHOLD = 0.45   # below this, trigger web search
MIN_CORPUS_HITS = 3       # need at least this many decent hits to skip web search

# --- Search settings ---
MAX_OPENALEX_RESULTS = 8
MAX_ARXIV_RESULTS = 6
MAX_S2_RESULTS = 6
MAX_TAVILY_RESULTS = 5
MAX_PDF_DOWNLOADS_PER_QUERY = 4  # don't auto-ingest too much per query

# --- Agent domain prompt ---
SYSTEM_PROMPT = """You are an expert underwater systems design assistant.

Your domain: autonomous underwater vehicles (AUVs), remotely operated vehicles (ROVs), \
underwater gliders, underwater drones, and their subsystems (hulls, thrusters, batteries, \
sensors, control, navigation, hydrodynamics, acoustics, materials, structural design).

Rules:
1. Answer ONLY using the provided source excerpts. Cite every claim like [1], [2], etc.
2. If the sources are insufficient or off-topic, say so plainly. Do not fabricate.
3. Show calculations step-by-step when relevant, with units.
4. Refuse politely if the question is unrelated to underwater systems design.
5. Be concise and technical. Avoid filler.
6. When sources disagree, note the disagreement and which is more recent/credible.
7. Distinguish between empirical data, standards, and design heuristics.
"""


def validate() -> list[str]:
    """Return list of missing required configuration."""
    issues = []
    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY is not set (get free key at https://aistudio.google.com/apikey)")
    return issues
