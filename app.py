"""Streamlit frontend for the Ollama RAG starter."""

from __future__ import annotations

import os
from typing import Optional
import html

import streamlit as st

from rag_pipeline import DEFAULT_CHROMA_DIR, RAGPipeline

st.set_page_config(page_title="Literature Library Retrieval", page_icon="🤖")

# Force a white background for the app
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff !important; }
    .stApp header { background-color: #ffffff !important; }
    .stAppViewContainer { background-color: #ffffff !important; }
    /* Title color to black */
    h1, .stMarkdown h1 { color: #000000 !important; }
    .block-container h1 { color: #000000 !important; }
    /* Make the question textarea gray */
    div[data-testid="stTextArea"] textarea {
        background-color: #f2f2f2 !important;
        color: #000000 !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder { color: #666666 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline(
    persist_directory: str = str(DEFAULT_CHROMA_DIR),
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> RAGPipeline:
    return RAGPipeline.from_chroma(
        persist_directory=os.path.abspath(persist_directory),
        embedding_model=embedding_model,
        llm_model=llm_model,
    )


st.title("Literature Library Retrieval")

with st.sidebar:
    st.header("Settings")
    persist_dir = st.text_input(
        "Chroma directory",
        value=str(DEFAULT_CHROMA_DIR),
        help="Path where the vector store is persisted",
    )
    llm_model = st.text_input(
        "LLM model",
        value=os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b"),
        help="Model name available in your local Ollama instance",
    )
    top_k = st.slider("Top K Documents", min_value=1, max_value=10, value=4)
    per_paper_max = st.number_input(
        "Max chunks per paper (0 = unlimited)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Limit how many chunks per paper can appear in the retrieved context; 0 means no limit.",
    )
    single_paper = st.checkbox(
        "Single paper only (top by RRF)",
        value=False,
        help="Select only the single most relevant paper by fused score and use its top-k chunks.",
    )
    paper_id = st.text_input(
        "Restrict to paper_id",
        value="",
        help="Optional. Use the slugified filename without extension (e.g., 'lupin' or 'predictive-business-process-monitoring').",
    )

question = st.text_area("Question", placeholder="What is this project about?", height=150)
submit = st.button("Generate Answer", type="primary")

if submit:
    if not question.strip():
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("Querying vector store and generating answer..."):
        try:
            pipeline = load_pipeline(
                persist_directory=persist_dir,
                llm_model=llm_model,
            )
            pipeline.retriever.search_kwargs["k"] = top_k
            # Wire UI controls into retriever behavior
            if single_paper:
                pipeline.retriever.search_kwargs["single_paper"] = True
            else:
                pipeline.retriever.search_kwargs.pop("single_paper", None)
            if int(per_paper_max) > 0:
                pipeline.retriever.search_kwargs["per_paper_max"] = int(per_paper_max)
            else:
                pipeline.retriever.search_kwargs.pop("per_paper_max", None)
            pid = paper_id.strip() or None
            result = pipeline.answer(question, paper_id=pid)
        except FileNotFoundError:
            st.error(
                "Chroma database not found. Run `python ingest.py` to create it first."
            )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
        else:
            st.subheader("Answer")
            # Force answer text to black using a dedicated class
            st.markdown(
                f"<p class='answer-text'>{html.escape(result['answer'])}</p>",
                unsafe_allow_html=True,
            )

            sources = result.get("sources", [])
            if sources:
                st.subheader("Sources")
                # Choose the majority paper among retrieved chunks to reflect main evidence
                counts = {}
                for s in sources:
                    md = s.get("metadata", {}) or {}
                    key = md.get("paper_id") or md.get("source") or "unknown"
                    counts[key] = counts.get(key, 0) + 1
                # Pick the key with the highest count; tie breaks by rank (earliest occurrence)
                best_key = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
                chosen = None
                if best_key is not None:
                    for s in sources:
                        md = s.get("metadata", {}) or {}
                        key = md.get("paper_id") or md.get("source") or "unknown"
                        if key == best_key:
                            chosen = s
                            break
                if chosen is None:
                    chosen = sources[0]

                metadata = chosen.get("metadata", {})
                st.markdown(
                    f"`{metadata.get('source', 'unknown')}`  "
                    f"(paper_id: `{metadata.get('paper_id', 'n/a')}`, section: `{metadata.get('section', 'n/a')}`)"
                )
                st.caption(chosen.get("content", ""))
