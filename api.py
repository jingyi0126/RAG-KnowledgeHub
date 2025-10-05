"""FastAPI service exposing the RAG pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import DEFAULT_CHROMA_DIR, RAGPipeline, generate_answer

app = FastAPI(title="RAG API", version="0.1.0")


class Source(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user question to answer")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Override how many documents to retrieve",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Override the Ollama chat model for this request",
    )
    paper_id: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a specific paper_id (derived from filename)",
    )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_chroma(persist_directory=DEFAULT_CHROMA_DIR)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    try:
        if request.llm_model or request.top_k:
            search_kwargs = {"k": request.top_k} if request.top_k else None
            result = generate_answer(
                request.question,
                llm_model=request.llm_model,
                search_kwargs=search_kwargs,
                paper_id=request.paper_id,
            )
        else:
            pipeline = get_pipeline()
            if request.top_k:
                pipeline.retriever.search_kwargs["k"] = request.top_k
            result = pipeline.answer(request.question, paper_id=request.paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(**result)
