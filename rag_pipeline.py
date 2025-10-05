"""Composable Retrieval-Augmented Generation pipeline powered by Ollama."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Any, Dict, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
DEFAULT_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()  # vector | hybrid

_DEFAULT_PROMPT = PromptTemplate.from_template(
    """You are an academic literature reading assistant. Answer the user's question strictly based on the "Context" below. The context is retrieved from the user's private paper library and may contain passages from one or more papers.

Requirements:
- Use only the provided context; do not invent or rely on external knowledge. If the context is insufficient, reply exactly: "I don't know based on the provided context".
- Write in concise English as a single cohesive paragraph (3–6 sentences). Do not use bullet points or section headings.
- You may include short quotations from the context using double quotes ("..."); do not fabricate citation numbers. The system will attach the actual sources and metadata after your answer.
- Only if there are significant gaps that prevent answering fully, append one final sentence starting with "Note:" that briefly states what is missing; otherwise do not mention uncertainty.

Context:
{context}

Question: {question}
"""
)


def build_embeddings(model: Optional[str] = None) -> OllamaEmbeddings:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = model or DEFAULT_EMBED_MODEL
    return OllamaEmbeddings(model=model_name, base_url=base_url)


def load_vector_store(
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    *,
    embedding_model: Optional[str] = None,
) -> Chroma:
    embeddings = build_embeddings(embedding_model)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def _load_all_docs_from_chroma(vector_store: Chroma, page_size: int = 1000) -> List[Document]:
    """Load all stored documents from Chroma to build a sparse (BM25) index.

    Uses the underlying collection API for pagination. Falls back to empty on failure.
    """
    docs: List[Document] = []
    col = getattr(vector_store, "_collection", None)
    if col is None:
        return docs
    offset = 0
    while True:
        try:
            res = col.get(
                include=["documents", "metadatas"],
                limit=page_size,
                offset=offset,
            )
        except Exception:
            break
        ids = res.get("ids", []) if isinstance(res, dict) else []
        documents = res.get("documents", []) if isinstance(res, dict) else []
        metadatas = res.get("metadatas", []) if isinstance(res, dict) else []
        if not ids:
            break
        for i, text in enumerate(documents):
            if not isinstance(text, str) or not text.strip():
                continue
            md = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            docs.append(Document(page_content=text, metadata=md))
        if len(ids) < page_size:
            break
        offset += page_size
    return docs


class HybridRRFRetriever(BaseRetriever):
    """Combine a dense retriever (Chroma) and a sparse retriever (BM25) using RRF.

    - Honors search_kwargs: {"k": int, "filter": {"paper_id": str}}
    - For BM25, applies metadata filter by post-filtering retrieved docs.
    - For vector, forwards the filter directly (Chroma supports it).
    """

    # Pydantic model fields (required by BaseRetriever which is a Pydantic model)
    vector_retriever: BaseRetriever
    bm25_retriever: Optional[BM25Retriever] = None
    weights: Tuple[float, float] = (0.5, 0.5)
    rrf_k: int = 60
    # Allow external control like other retrievers (k, filter, etc.)
    search_kwargs: Dict[str, Any] = {}

    def _doc_key(self, d: Any) -> str:
        md = getattr(d, "metadata", {}) or {}
        # Prefer stable id if present
        return str(md.get("chunk_sha1") or md.get("source") or id(d))

    def _apply_filter(self, docs: List[Any], flt: Optional[Dict[str, Any]]) -> List[Any]:
        if not flt:
            return docs
        # Only implement paper_id filter for now
        pid = flt.get("paper_id") if isinstance(flt, dict) else None
        if pid is None:
            return docs
        out: List[Any] = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            if md.get("paper_id") == pid:
                out.append(d)
        return out

    def _get_relevant_documents(self, query: str) -> List[Any]:
        k = int(self.search_kwargs.get("k", DEFAULT_K))
        # Optional controls
        per_paper_max = int(self.search_kwargs.get("per_paper_max", 0) or 0)
        single_paper_only = bool(self.search_kwargs.get("single_paper", False))
        flt = self.search_kwargs.get("filter")

        # Dense results (forward filter)
        dense_docs: List[Any] = []
        try:
            # Some retrievers accept search_kwargs directly; otherwise set attr then call
            if hasattr(self.vector_retriever, "search_kwargs") and isinstance(self.vector_retriever.search_kwargs, dict):
                old_filter = self.vector_retriever.search_kwargs.get("filter")
                try:
                    if flt is not None:
                        self.vector_retriever.search_kwargs["filter"] = flt
                    dense_docs = self.vector_retriever.get_relevant_documents(query)
                finally:
                    if flt is not None:
                        if old_filter is not None:
                            self.vector_retriever.search_kwargs["filter"] = old_filter
                        else:
                            self.vector_retriever.search_kwargs.pop("filter", None)
            else:
                dense_docs = self.vector_retriever.get_relevant_documents(query)
        except Exception:
            dense_docs = []

        # Sparse results (BM25), then post-filter
        sparse_docs: List[Any] = []
        if self.bm25_retriever is not None:
            try:
                sparse_docs = self.bm25_retriever.get_relevant_documents(query)
                sparse_docs = self._apply_filter(sparse_docs, flt)
            except Exception:
                sparse_docs = []

        # RRF fusion
        # rank per list
        scores: Dict[str, float] = {}
        by_key: Dict[str, Any] = {}

        def add_list(lst: List[Any], weight: float) -> None:
            for rank, d in enumerate(lst, start=1):
                key = self._doc_key(d)
                by_key.setdefault(key, d)
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (self.rrf_k + rank))

        add_list(dense_docs, self.weights[0])
        add_list(sparse_docs, self.weights[1])

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        # Helper to extract paper key
        def paper_key(d: Any) -> str:
            md = getattr(d, "metadata", {}) or {}
            return str(md.get("paper_id") or md.get("source") or "unknown")

        # If single_paper_only: select the best paper by summed score, then keep its top docs only
        if single_paper_only and ranked:
            paper_scores: Dict[str, float] = {}
            for key, sc in ranked:
                pkey = paper_key(by_key[key])
                paper_scores[pkey] = paper_scores.get(pkey, 0.0) + sc
            # Best paper id
            best_paper = max(paper_scores.items(), key=lambda kv: kv[1])[0]
            filtered = [(key, sc) for key, sc in ranked if paper_key(by_key[key]) == best_paper]
            return [by_key[key] for key, _ in filtered[:k]] or (dense_docs[:k] if dense_docs else sparse_docs[:k])

        # Else optionally cap per-paper occurrences while preserving global rank order
        if per_paper_max > 0 and ranked:
            kept: List[Any] = []
            counts: Dict[str, int] = {}
            for key, _ in ranked:
                d = by_key[key]
                pkey = paper_key(d)
                if counts.get(pkey, 0) >= per_paper_max:
                    continue
                kept.append(d)
                counts[pkey] = counts.get(pkey, 0) + 1
                if len(kept) >= k:
                    break
            if kept:
                return kept

        # Default: take top-k after RRF
        docs = [by_key[key] for key, _ in ranked[:k]]
        if not docs:
            docs = dense_docs[:k] if dense_docs else sparse_docs[:k]
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Any]:
        # Simple sync wrapper
        return self._get_relevant_documents(query)


def build_retriever(
    vector_store: Chroma,
    *,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    kwargs = {"k": DEFAULT_K}
    if search_kwargs:
        kwargs.update(search_kwargs)

    # Determine mode
    mode = (kwargs.pop("mode", None) or DEFAULT_RETRIEVAL_MODE).lower()

    # Always build the vector retriever
    # Prefer MMR to improve diversity and reduce near-duplicate chunks
    stype = os.getenv("RETRIEVAL_VECTOR_SEARCH_TYPE", "mmr").lower()  # "mmr" | "similarity"
    vkwargs = dict(kwargs)
    if stype == "mmr":
        # How many candidates to fetch before MMR reranking; default 4x k
        k_val = int(vkwargs.get("k", DEFAULT_K))
        fetch_k = int(os.getenv("MMR_FETCH_K", str(k_val * 4)))
        lambda_mult = float(os.getenv("MMR_LAMBDA", "0.5"))
        vkwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
    # Build the retriever with desired search type
    vector_retriever = vector_store.as_retriever(
        search_type=stype if stype in {"mmr", "similarity"} else "similarity",
        search_kwargs=vkwargs,
    )

    if mode != "hybrid":
        return vector_retriever

    # Build BM25 corpus from Chroma docs
    bm25_docs = _load_all_docs_from_chroma(vector_store)
    bm25_ret: Optional[BM25Retriever] = None
    if bm25_docs:
        try:
            bm25_ret = BM25Retriever.from_documents(bm25_docs)
            bm25_ret.k = kwargs.get("k", DEFAULT_K)  # type: ignore[attr-defined]
        except Exception:
            bm25_ret = None

    # Hybrid retriever with RRF
    # Configure RRF weights (dense, sparse) and k
    weights = (0.7, 0.3)
    weights_env = os.getenv("RRF_WEIGHTS", "").strip()
    if weights_env:
        try:
            parts = [float(x) for x in weights_env.split(",")]
            if len(parts) == 2 and all(p >= 0 for p in parts):
                weights = (parts[0], parts[1])
        except Exception:
            pass
    return HybridRRFRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_ret,
        weights=weights,
        rrf_k=int(os.getenv("RRF_K", "60")),
        search_kwargs={"k": kwargs.get("k", DEFAULT_K)},
    )


def build_llm(model: Optional[str] = None) -> ChatOllama:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = model or DEFAULT_LLM_MODEL
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)


@dataclass
class RAGPipeline:
    """Thin wrapper bundling retriever and LLM into a ready-to-use pipeline."""

    retriever: BaseRetriever
    qa_chain: RetrievalQA

    @classmethod
    def from_chroma(
        cls,
        *,
        persist_directory: Path = DEFAULT_CHROMA_DIR,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        prompt: PromptTemplate = _DEFAULT_PROMPT,
        search_kwargs: Optional[dict] = None,
    ) -> "RAGPipeline":
        vector_store = load_vector_store(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )
        retriever = build_retriever(vector_store, search_kwargs=search_kwargs)
        llm = build_llm(llm_model)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return cls(retriever=retriever, qa_chain=qa_chain)

    def retrieve(self, query: str) -> List[str]:
        docs = self.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]

    def answer(self, question: str, *, paper_id: Optional[str] = None) -> dict:
        """Run the QA chain and return the answer plus sources."""
        # Temporarily apply a metadata filter if provided
        old_filter: Optional[Dict[str, Any]] = self.retriever.search_kwargs.get("filter")  # type: ignore[attr-defined]
        try:
            if paper_id:
                self.retriever.search_kwargs["filter"] = {"paper_id": paper_id}  # type: ignore[attr-defined]
            result = self.qa_chain({"query": question})
        finally:
            # Restore previous filter (including removing ours if none before)
            if paper_id is not None:
                if old_filter is not None:
                    self.retriever.search_kwargs["filter"] = old_filter  # type: ignore[attr-defined]
                else:
                    self.retriever.search_kwargs.pop("filter", None)  # type: ignore[attr-defined]
        # LangChain returns a dict with keys {"result", "source_documents"}
        return {
            "answer": result["result"],
            "sources": [
                {
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                }
                for doc in result.get("source_documents", [])
            ],
        }


def generate_answer(
    question: str,
    *,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    search_kwargs: Optional[dict] = None,
    paper_id: Optional[str] = None,
) -> dict:
    pipeline = RAGPipeline.from_chroma(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        llm_model=llm_model,
        search_kwargs=search_kwargs,
    )
    return pipeline.answer(question, paper_id=paper_id)
