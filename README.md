# Local RAG Playground (Ollama + Chroma)

A local retrieval‑augmented generation (RAG) system for reading and querying your own academic papers. It ingests PDFs/TXT/MD files, cleans and chunks them, builds embeddings with Ollama, stores vectors in Chroma, and answers questions in a Streamlit UI or via a FastAPI endpoint.

> Goal: Answer questions strictly based on your local papers with concise, single‑paragraph responses and clear provenance.

---

## What this project does

- Ingests documents from `data/` and persists a local Chroma vector store under `chroma_db/`.
- Cleans academic PDFs (two‑column aware), splits by sections and then into overlapping text chunks.
- Embeds text using a local Ollama embedding model (default: `nomic-embed-text`).
- Retrieves with hybrid sparse+dense search and Reciprocal Rank Fusion (RRF).
- Generates answers with a local Ollama chat model (default: `llama3:8b`).
- Provides:
  - Streamlit UI (`app.py`) for interactive Q&A.
  - FastAPI endpoint (`api.py`) for programmatic access.

---

## Architecture and methods

### 1) Ingestion pipeline (`ingest.py`)

- Loaders
  - PDFs: PyMuPDF (fitz) with a simple two‑column heuristic; falls back to `PyPDFLoader` when PyMuPDF is unavailable.
  - Text: `TextLoader` for `.txt` and `.md`.
- Academic PDF cleaning and normalization
  - Merge hyphenated words across line breaks (e.g., `gener-\nalization` → `generalization`).
  - Normalize line breaks; collapse soft wraps; trim excessive whitespace.
  - Heuristically remove common headers/footers/page numbers.
  - Keep from Abstract to before References/Bibliography/Acknowledgments when present.
  - Split by structural headings (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion, or numbered headings like `1.`, `2.1`...).
- Chunking
  - Section‑first, then character chunking via `RecursiveCharacterTextSplitter`.
  - Defaults: `chunk_size=800`, `chunk_overlap=150` (tuned for academic prose).
- Metadata and deduplication
  - `paper_id`: slugified filename (e.g., `LUPIN.pdf` → `lupin`).
  - `section`: section title or `body`.
  - `chunk_sha1`: SHA‑1 hash prefix of chunk text.
  - Deterministic chunk IDs: `paper_id:section:chunk_sha1`.
  - Dedup: precheck existing IDs in Chroma; skip already‑ingested chunks.
- Embeddings and persistence
  - Embedding model: Ollama (`nomic-embed-text` by default) at `OLLAMA_BASE_URL` (`http://localhost:11434`).
  - Fast‑fail connectivity check (`embed_query('healthcheck')`).
  - Batched `add_documents` with periodic `persist()` and progress logging; batch size is configurable (default 64).

### 2) Retrieval pipeline (`rag_pipeline.py`)

- Vector store: Chroma (local persistence)
- Dense retriever: Chroma retriever (supports metadata filter, e.g., by `paper_id`).
- Sparse retriever: `BM25Retriever` built from all Chroma docs.
- Fusion: Reciprocal Rank Fusion (RRF)
  - Combine dense and sparse lists with weights and `rrf_k`.
  - Honors `search_kwargs` (e.g., `{k, filter}`), and applies `paper_id` filtering.
- Prompting and generation
  - Prompt requires: single concise English paragraph (3–6 sentences), strictly context‑grounded; reply with "I don't know based on the provided context" if insufficient.
  - Optional final "Note:" sentence only when important gaps remain.
  - Chat model: Ollama `llama3:8b` by default, temperature 0.

### 3) Frontends

- Streamlit UI (`app.py`)
  - Sidebar settings: `persist_dir`, `llm_model`, `top_k`, optional `paper_id` filter.
  - Displays answer and sources; sources are deduplicated in the UI by `paper_id` (fallback `source`) so only one reference is shown.
- FastAPI (`api.py`)
  - `POST /query` with fields: `question`, `top_k`, `llm_model`, optional `paper_id`.

---

## Repository layout

- `ingest.py` — Data ingestion, cleaning, chunking, embeddings, and Chroma persistence (with dedup and progress batching).
- `rag_pipeline.py` — Embeddings + Chroma loader, hybrid RRF retriever, prompt, and a thin RAG pipeline.
- `app.py` — Streamlit UI for interactive Q&A with deduplicated source display.
- `api.py` — FastAPI server exposing `/query`.
- `data/` — Place your PDFs, `.txt`, and `.md` files here.
- `chroma_db/` — Local Chroma persistent store (created after ingest).
- `scripts/inspect_chroma.py` — Utility to inspect what’s inside Chroma (counts by paper/source).
- `requirements.txt` — Python dependencies.

---

## Setup

- Requirements
  - Python 3.10–3.12 recommended
  - Ollama installed and running locally
  - NVIDIA GPU optional but recommended

- Install Python dependencies (in a virtual environment)

  Optional commands:
  ```powershell
  # Windows PowerShell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- Install/pull Ollama models (embedding + chat)

  Optional commands:
  ```powershell
  ollama pull nomic-embed-text
  ollama pull llama3:8b
  ollama list
  ```

> Note on protobuf: Some Streamlit versions expect `protobuf<6`, while recent chromadb may bring protobuf>=6. If Streamlit errors mention protobuf, consider pinning `protobuf<6` and reinstalling Streamlit, or using a separate environment for the UI.

---

## Ingest your data

Optional commands:
```powershell
# Basic ingest (uses data/ and writes to chroma_db/)
python ingest.py

# Preview chunking only (no embeddings/DB writes)
python ingest.py --preview

# Ingest a single file (substring match)
python ingest.py --only-file "LUPIN.pdf"
```

Environment variables (optional):
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`)
- `INGEST_CHUNK_SIZE` (default `800`)
- `INGEST_CHUNK_OVERLAP` (default `150`)
- `INGEST_BATCH_SIZE` (default `64`)

If it looks "stuck" at first embed: ensure Ollama is running and models are pulled. You can check:

Optional commands:
```powershell
Invoke-WebRequest -Uri http://localhost:11434/api/tags -UseBasicParsing # 200 OK if healthy
ollama ps  # see active operations
```

---

## Run the UI

Optional commands:
```powershell
streamlit run app.py
```

- Sidebar lets you set `persist_dir`, `llm_model`, `top_k`, and `paper_id`.
- Sources are deduplicated by `paper_id`/`source` to show only one reference per answer.

---

## Run the API

Optional commands:
```powershell
# If uvicorn is installed
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

`POST /query`
- JSON: `{ "question": str, "top_k": int (optional), "llm_model": str (optional), "paper_id": str (optional) }`
- Returns: `{ "answer": str, "sources": [{ metadata, content }] }`

---

## Configuration

Key environment variables:
- `OLLAMA_BASE_URL` — Ollama HTTP base URL
- `OLLAMA_CHAT_MODEL` — chat model name (default: `llama3:8b`)
- `OLLAMA_EMBED_MODEL` — embedding model name (default: `nomic-embed-text`)
- `RETRIEVAL_MODE` — `vector` | `hybrid` (default: `hybrid`)
- `RETRIEVAL_TOP_K` — default `k` for retrieval
- `RRF_K` — RRF smoothing constant (default: `60`)
- `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `INGEST_BATCH_SIZE`

You can also control `top_k` and `paper_id` at runtime from the UI or API.

---

## Troubleshooting

- Model not found / 404
  - Pull the model: `ollama pull nomic-embed-text` or adjust env/model name to one you have.
- Connection refused to `localhost:11434`
  - Start Ollama desktop/service; verify with `/api/tags`.
- Ingest appears stuck
  - First run may download/compile models; pre‑pull to avoid silent waits.
  - This project batches adds and logs progress; you can reduce `INGEST_BATCH_SIZE` for more frequent updates.
- Streamlit + protobuf error
  - Consider pinning `protobuf<6` or isolating the UI in a separate environment.
- Deprecation warnings (`langchain_community.vectorstores.Chroma`, `OllamaEmbeddings`)
  - Safe to ignore short‑term. Future migration path: `langchain-chroma` and `langchain-ollama`.

---

## Design choices and rationale

- Section‑aware PDF processing improves semantic coherence of chunks.
- Deterministic IDs enable safe re‑ingest and dedup.
- Hybrid retrieval (BM25 + vectors) with RRF often yields better recall/precision than either alone.
- UI deduplicates displayed sources so you see one clear reference, even if multiple chunks were used internally.
- Prompt enforces grounded, single‑paragraph answers and an explicit "I don't know" fallback to reduce hallucinations.

---

## Roadmap (optional)

- Migrate to `langchain-chroma` and `langchain-ollama` packages.
- Add unit tests and evaluation harness for retrieval quality.
- Optional per‑paper reranker or "max one chunk per paper" retrieval policy.

---

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
