"""Streamlit web UI for the underwater design agent.

Run with:  streamlit run src/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src import config
from src.agent import Agent, AgentTrace

# ---------- Page setup ----------

st.set_page_config(
    page_title="Underwater Design Agent",
    page_icon="🌊",
    layout="wide",
)


# ---------- Config check ----------

issues = config.validate()
if issues:
    st.error("Configuration problems:")
    for i in issues:
        st.write(f"- {i}")
    st.info("Edit `.env` and restart Streamlit.")
    st.stop()


# ---------- Agent (cached so we don't reload Qdrant on every interaction) ----------

@st.cache_resource(show_spinner="Loading agent (first time may take ~30s for embedding model)...")
def get_agent():
    return Agent()


agent = get_agent()


# ---------- Sidebar ----------

with st.sidebar:
    st.title("🌊 Underwater Agent")
    st.caption("AUVs · ROVs · Underwater Drones · Marine Robotics")

    st.divider()
    st.subheader("Corpus")
    n_chunks = agent.corpus.count()
    n_sources = agent.corpus.count_sources()
    col1, col2 = st.columns(2)
    col1.metric("Chunks", n_chunks)
    col2.metric("Sources", n_sources)

    st.caption(
        "The corpus grows automatically as you ask questions. "
        "Open-access papers found via web search are downloaded and indexed."
    )

    st.divider()
    st.subheader("Sources searched")
    st.markdown(
        "- OpenAlex (250M+ works)\n"
        "- arXiv\n"
        "- Semantic Scholar\n"
        "- Crossref\n"
        "- Unpaywall (OA PDF resolver)\n"
        + ("- Tavily web search\n" if config.TAVILY_API_KEY else "- _Tavily disabled (no key)_\n")
    )

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ---------- Main chat ----------

st.title("Ask anything about underwater systems design")
st.caption(f"Model: `{config.GEMINI_MODEL}`  ·  Embeddings: `{config.EMBEDDING_MODEL}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace"):
            _render_trace = m["trace"]
            with st.expander(f"🔍 Trace ({len(_render_trace.get('sources_used', []))} sources)"):
                for n in _render_trace.get("notes", []):
                    st.caption(n)
                for src in _render_trace.get("sources_used", []):
                    st.markdown(
                        f"**[{src['index']}]** {src['title']}  \n"
                        f"_{src.get('authors', '')} · {src.get('year', '') or 'n.d.'} · {src['type']}_  \n"
                        f"[link]({src['url']})"
                    )


# New input
question = st.chat_input("e.g. 'What hull shapes are used for survey-class AUVs and why?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.status("Thinking...", expanded=False)
        accumulated = ""
        last_trace: AgentTrace | None = None

        for piece, trace in agent.answer(question):
            accumulated += piece
            placeholder.markdown(accumulated + "▌")
            last_trace = trace
            # Update status line
            status.update(label=f"{trace.notes[-1] if trace.notes else 'Working...'}")

        placeholder.markdown(accumulated)
        status.update(label="Done", state="complete")

        # Show trace expander
        if last_trace:
            with st.expander(f"🔍 Trace ({len(last_trace.sources_used)} sources used)"):
                for n in last_trace.notes:
                    st.caption(n)
                st.divider()
                for src in last_trace.sources_used:
                    st.markdown(
                        f"**[{src['index']}]** {src['title']}  \n"
                        f"_{src.get('authors', '')} · {src.get('year', '') or 'n.d.'} · {src['type']}_  \n"
                        f"[link]({src['url']})"
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": accumulated,
            "trace": {
                "notes": last_trace.notes if last_trace else [],
                "sources_used": last_trace.sources_used if last_trace else [],
            } if last_trace else None,
        })
