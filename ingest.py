"""Data ingestion script for populating the local Chroma vector store.

This script walks the ``data`` directory, loads supported documents, splits them
into semantic chunks, embeds them with Ollama, and persists them into a Chroma
vector database located under ``chroma_db``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import re
from collections import defaultdict
import json
import hashlib

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Tuned for academic PDFs per user requirement
DEFAULT_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "150"))
DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))


def iter_document_paths(data_dir: Path) -> Iterable[Path]:
    """Yield supported document paths under ``data_dir`` recursively."""
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_documents(data_dir: Path, *, only_file: Optional[str] = None) -> List[Document]:
    """Load all supported documents from ``data_dir`` into LangChain documents.

    only_file: optional substring filter (case-insensitive) on filename.
    """
    docs: List[Document] = []
    for file_path in iter_document_paths(data_dir):
        if only_file:
            if only_file.lower() not in file_path.name.lower():
                continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            # Prefer PyMuPDF-based extraction for better two-column handling when available
            loaded_docs: List[Document]
            try:
                loaded_docs = _load_pdf_twocol_pymupdf(file_path)
            except Exception as e:  # noqa: BLE001
                LOGGER.warning(
                    "PyMuPDF extraction failed or unavailable for %s (%s). Falling back to PyPDFLoader.",
                    file_path.name,
                    e,
                )
                loader = PyPDFLoader(str(file_path))
                loaded_docs = loader.load()
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded_docs = loader.load()
        LOGGER.info("Loaded %s (%d pages)", file_path.name, len(loaded_docs))
        docs.extend(loaded_docs)
    return docs


def _load_pdf_twocol_pymupdf(file_path: Path) -> List[Document]:
    """Extract PDF text with two-column awareness using PyMuPDF (fitz).

    Strategy per page:
    - Get text blocks with bounding boxes
    - Split into left/right by page mid-x (simple heuristic)
    - Sort each column by y (top->bottom)
    - Concatenate left column then right column
    Returns page-level Documents with metadata {source, page}.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyMuPDF (pymupdf) not installed") from exc

    docs: List[Document] = []
    with fitz.open(str(file_path)) as pdf:
        for page_index, page in enumerate(pdf):
            page_rect = page.rect
            mid_x = (page_rect.x0 + page_rect.x1) / 2.0
            try:
                blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1, text, block_no, ...)
            except Exception:
                # fallback to simple text
                text = page.get_text("text")
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(file_path), "page": page_index},
                    )
                )
                continue

            left_blocks = []
            right_blocks = []
            for b in blocks:
                if len(b) < 5:
                    continue
                x0, y0, x1, y1, text = b[:5]
                if not isinstance(text, str) or not text.strip():
                    continue
                (left_blocks if x0 < mid_x else right_blocks).append((y0, x0, text))

            # Determine if single-column (one side empty), then just merge all by y,x
            def sort_and_join(blks: List[Tuple[float, float, str]]) -> str:
                return "\n".join(t for _, __, t in sorted(blks, key=lambda z: (z[0], z[1])))

            if left_blocks and right_blocks:
                left_text = sort_and_join(left_blocks)
                right_text = sort_and_join(right_blocks)
                page_text = left_text + "\n\n" + right_text
            else:
                page_text = sort_and_join(left_blocks + right_blocks)

            docs.append(
                Document(
                    page_content=page_text,
                    metadata={"source": str(file_path), "page": page_index},
                )
            )
    return docs


def _merge_hyphenated_words(text: str) -> str:
    """Merge words broken by hyphen at line end (e.g., "gener-\nalization" -> "generalization")."""
    return re.sub(r"-\n(?=[a-z])", "", text)


def _normalize_linebreaks(text: str) -> str:
    """Normalize line endings and collapse soft wraps to improve chunk quality."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)  # limit consecutive newlines
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)  # soft wrap -> space
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t


def _strip_headers_footers_lines(lines: List[str]) -> List[str]:
    """Heuristically remove typical page headers/footers and page numbers."""
    cleaned: List[str] = []
    for ln in lines:
        l = ln.strip()
        if not l:
            continue
        if re.fullmatch(r"\d{1,3}", l) or re.fullmatch(r"(?i)page\s+\d{1,4}", l):
            continue
        if re.search(r"(?i)arxiv preprint|proceedings|cc by|copyright|doi:|issn", l):
            continue
        cleaned.append(ln)
    return cleaned


def _keep_abstract_to_body(text: str) -> str:
    """Keep content from Abstract to before References/Bibliography/Acknowledgments when present."""
    lower = text.lower()
    start = None
    for token in ["abstract", "summary"]:
        m = re.search(rf"\b{token}\b\s*:?(\s|\n)", lower)
        if m:
            start = m.start()
            break
    if start is not None:
        text = text[start:]
        lower = text.lower()

    end_match = re.search(r"\b(references|bibliography|acknowledg?ments?)\b", lower)
    if end_match:
        text = text[: end_match.start()]
    return text.strip()


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split academic text by structural headings. Returns list of (section_title, section_text)."""
    raw_lines = [ln.rstrip() for ln in text.split("\n")]
    lines = _strip_headers_footers_lines(raw_lines)

    heading_pat = re.compile(
        r"^(?P<h>(abstract|introduction|background|related work|method|methods|approach|methodology|model|experiments|results|evaluation|discussion|conclusion|conclusions|future work|limitations)|\d+(?:\.\d+)*\.?\s+.+?)\s*$",
        re.IGNORECASE,
    )

    sections: List[Tuple[str, List[str]]] = []
    current_title = "Abstract/Body"
    current_lines: List[str] = []

    for ln in lines:
        m = heading_pat.match(ln)
        if m:
            if current_lines:
                sections.append((current_title, current_lines))
                current_lines = []
            current_title = m.group("h").strip()
        else:
            current_lines.append(ln)

    if current_lines:
        sections.append((current_title, current_lines))

    out: List[Tuple[str, str]] = []
    for title, sec_lines in sections:
        sec_text = "\n".join(sec_lines)
        sec_text = _normalize_linebreaks(_merge_hyphenated_words(sec_text))
        if sec_text:
            out.append((title, sec_text))
    return out


def clean_pdf_documents(page_docs: List[Document]) -> List[Document]:
    """Group page-level PDF docs by source and clean: keep Abstract+Body, drop headers/footers/refs, split into sections."""
    by_src: Dict[str, List[Document]] = defaultdict(list)
    for d in page_docs:
        src = d.metadata.get("source") or "unknown"
        by_src[src].append(d)

    cleaned_docs: List[Document] = []
    for src, docs in by_src.items():
        docs_sorted = sorted(docs, key=lambda x: x.metadata.get("page", 0))
        full_text = "\n".join(d.page_content for d in docs_sorted)

        text = _merge_hyphenated_words(full_text)
        text = _keep_abstract_to_body(text)
        text = _normalize_linebreaks(text)

        sections = _split_into_sections(text)
        if not sections:
            cleaned_docs.append(
                Document(page_content=text, metadata={"source": src, "section": "body"})
            )
            continue
        for title, sec_text in sections:
            meta = {"source": src, "section": title}
            cleaned_docs.append(Document(page_content=sec_text, metadata=meta))

    return cleaned_docs

def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding.

    For PDF academic papers, perform cleaning and section-based splitting first
    to better preserve semantics, then apply character chunking.
    """
    docs_list = list(documents)
    pdf_page_docs = [
        d for d in docs_list if str(d.metadata.get("source", "")).lower().endswith(".pdf")
    ]
    other_docs = [d for d in docs_list if d not in pdf_page_docs]

    processed_docs: List[Document] = []
    if pdf_page_docs:
        processed_docs.extend(clean_pdf_documents(pdf_page_docs))
    if other_docs:
        for d in other_docs:
            txt = _normalize_linebreaks(_merge_hyphenated_words(d.page_content))
            md = dict(d.metadata)
            src = str(md.get("source", "unknown"))
            base = os.path.splitext(os.path.basename(src))[0].lower()
            paper_id = re.sub(r"[^a-z0-9]+", "-", base).strip("-") if src != "unknown" else "unknown"
            md.setdefault("paper_id", paper_id)
            md.setdefault("section", md.get("section", "body"))
            processed_docs.append(Document(page_content=txt, metadata=md))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(processed_docs)


def preview_chunks(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_preview: int = 3,
    max_groups: int = 10,
    preview_out: Optional[Path] = None,
    preview_format: str = "jsonl",
    only_file: Optional[str] = None,
) -> None:
    """Load and split documents, then print stats and a few sample chunks without persisting."""
    documents = load_documents(data_dir, only_file=only_file)
    if not documents:
        raise ValueError("No documents found for preview.")

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        LOGGER.info("No chunks produced. Consider adjusting chunk parameters.")
        return

    LOGGER.info(
        "Total chunks: %d | Min: %d | Max: %d | Avg: %.1f",
        len(chunks),
        min(lengths),
        max(lengths),
        sum(lengths) / len(lengths),
    )

    # Group by source/section for a quick look
    by_key: Dict[Tuple[str, str], List[Document]] = defaultdict(list)
    for c in chunks:
        src = str(c.metadata.get("source", "unknown"))
        sec = str(c.metadata.get("section", "n/a"))
        by_key[(src, sec)].append(c)

    shown_groups = 0
    for (src, sec), docs in by_key.items():
        LOGGER.info("Source: %s | Section: %s | Chunks: %d", src, sec, len(docs))
        for i, d in enumerate(docs[:max_preview], start=1):
            preview = d.page_content[:200].replace("\n", " ")
            LOGGER.info(
                "  #%d len=%d: %s%s",
                i,
                len(d.page_content),
                preview,
                "..." if len(d.page_content) > 200 else "",
            )
        shown_groups += 1
        if shown_groups >= max_groups:
            break

    # Optionally export full preview to file
    if preview_out is not None:
        # Infer format from argument or file suffix
        fmt = preview_format.lower()
        suf = preview_out.suffix.lower()
        if fmt == "jsonl" or suf in {".jsonl", ".ndjson"}:
            fmt = "jsonl"
        elif fmt == "md" or suf == ".md":
            fmt = "md"
        elif fmt == "txt" or suf == ".txt":
            fmt = "txt"
        else:
            fmt = "jsonl"

        preview_out = Path(preview_out)
        preview_out.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "jsonl":
            with preview_out.open("w", encoding="utf-8") as f:
                header = {
                    "type": "summary",
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                f.write(json.dumps(header, ensure_ascii=False) + "\n")
                for (src, sec), docs in by_key.items():
                    for idx, d in enumerate(docs):
                        rec = {
                            "type": "chunk",
                            "source": src,
                            "section": sec,
                            "index": idx,
                            "length": len(d.page_content),
                            "content": d.page_content,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elif fmt == "md":
            with preview_out.open("w", encoding="utf-8") as f:
                f.write(f"# Chunk Preview\n\nTotal chunks: {len(chunks)}  ")
                f.write(f"Chunk size: {chunk_size}  Overlap: {chunk_overlap}\n\n")
                for (src, sec), docs in by_key.items():
                    f.write(f"## Source: {src}\n\n")
                    f.write(f"### Section: {sec} (Chunks: {len(docs)})\n\n")
                    for idx, d in enumerate(docs, start=1):
                        f.write(f"- Chunk #{idx} (len={len(d.page_content)})\n\n")
                        f.write("```\n")
                        f.write(d.page_content)
                        f.write("\n```\n\n")
        else:  # txt
            with preview_out.open("w", encoding="utf-8") as f:
                f.write(
                    f"Total chunks: {len(chunks)} | size={chunk_size} overlap={chunk_overlap}\n\n"
                )
                for (src, sec), docs in by_key.items():
                    f.write(f"Source: {src} | Section: {sec} | Chunks: {len(docs)}\n")
                    for idx, d in enumerate(docs, start=1):
                        f.write(f"  #{idx} len={len(d.page_content)}\n")
                        f.write(d.page_content + "\n\n")
        LOGGER.info("Preview exported to %s (%s)", preview_out, fmt)


def build_embeddings(model: str | None = None) -> OllamaEmbeddings:
    """Create an Ollama embeddings client using environment configuration."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = model or DEFAULT_EMBED_MODEL
    LOGGER.info("Using Ollama embeddings model: %s", model_name)
    return OllamaEmbeddings(model=model_name, base_url=base_url)


def ingest(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    model: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    only_file: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Chroma:
    """Run the full ingestion pipeline and persist to Chroma."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = load_documents(data_dir, only_file=only_file)
    if not documents:
        raise ValueError(
            "No documents found. Add PDF, TXT, or MD files to the data directory."
        )

    chunks = split_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    LOGGER.info("Generated %d text chunks", len(chunks))

    # Enrich metadata with a stable paper_id and build deterministic chunk ids
    def _slugify(name: str) -> str:
        base = os.path.splitext(os.path.basename(name))[0].lower()
        return re.sub(r"[^a-z0-9]+", "-", base).strip("-")

    enriched: List[Document] = []
    ids: List[str] = []
    for d in chunks:
        src = str(d.metadata.get("source", "unknown"))
        paper_id = _slugify(src) if src != "unknown" else "unknown"
        content_hash = hashlib.sha1(d.page_content.encode("utf-8")).hexdigest()[:16]
        section = str(d.metadata.get("section", "body"))
        chunk_id = f"{paper_id}:{section}:{content_hash}"

        md = dict(d.metadata)
        md["paper_id"] = paper_id
        md["chunk_sha1"] = content_hash
        enriched.append(Document(page_content=d.page_content, metadata=md))
        ids.append(chunk_id)

    embeddings = build_embeddings(model)

    # Quick connectivity check to fail fast if Ollama isn't reachable or model isn't pulled
    try:
        _ = embeddings.embed_query("healthcheck")
        LOGGER.info("Verified Ollama embeddings endpoint is reachable.")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Unable to reach Ollama embeddings endpoint. Ensure 'ollama' is running, the model is pulled, and OLLAMA_BASE_URL is correct."
        ) from exc

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    # Deduplicate: check which ids already exist and only add new ones
    existing_ids: set[str] = set()
    try:
        # Try public get in batches
        B = 500
        for i in range(0, len(ids), B):
            batch = ids[i : i + B]
            res = vectordb.get(ids=batch)  # type: ignore[attr-defined]
            for rid in res.get("ids", []) if isinstance(res, dict) else []:
                if isinstance(rid, list):
                    existing_ids.update(rid)
                elif isinstance(rid, str):
                    existing_ids.add(rid)
    except Exception:
        # Fallback to private collection API if available
        try:
            col = getattr(vectordb, "_collection", None)
            if col is not None:
                B = 500
                for i in range(0, len(ids), B):
                    batch = ids[i : i + B]
                    res = col.get(ids=batch)
                    for rid in res.get("ids", []) if isinstance(res, dict) else []:
                        if isinstance(rid, list):
                            existing_ids.update(rid)
                        elif isinstance(rid, str):
                            existing_ids.add(rid)
        except Exception:
            LOGGER.warning("Could not prefetch existing ids; proceeding to add all (may duplicate).")

    to_add_docs: List[Document] = []
    to_add_ids: List[str] = []
    for doc, cid in zip(enriched, ids):
        if cid in existing_ids:
            continue
        to_add_docs.append(doc)
        to_add_ids.append(cid)

    if to_add_docs:
        total = len(to_add_docs)
        LOGGER.info(
            "Adding %d new chunks in batches of %d (skipping %d existing)",
            total,
            batch_size,
            len(enriched) - total,
        )
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_docs = to_add_docs[start:end]
            batch_ids = to_add_ids[start:end]
            try:
                vectordb.add_documents(batch_docs, ids=batch_ids)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed adding batch %d-%d: %s", start, end, exc)
                raise
            # Persist after each batch so progress is saved and resumable
            vectordb.persist()
            LOGGER.info("Progress: %d/%d chunks ingested (%.1f%%)", end, total, (end / total) * 100.0)
        LOGGER.info("Completed ingestion. Persisted Chroma DB to %s", persist_directory)
    else:
        LOGGER.info("No new chunks to add. Database is up-to-date at %s", persist_directory)
    return vectordb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Directory to store the Chroma database",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama embedding model name (overrides OLLAMA_EMBED_MODEL)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Number of overlapping characters between chunks",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview chunking stats and sample chunks without persisting",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=3,
        help="Max sample chunks to show per source/section",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=10,
        help="Max (source,section) groups to print in console",
    )
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="Write full preview to a file (.jsonl/.md/.txt)",
    )
    parser.add_argument(
        "--preview-format",
        type=str,
        choices=["jsonl", "md", "txt"],
        default="jsonl",
        help="Preview export format when --preview-out is provided",
    )
    parser.add_argument(
        "--only-file",
        type=str,
        default=None,
        help="Preview or ingest only files whose name contains this substring (case-insensitive)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if args.preview:
        preview_chunks(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_preview=args.max_preview,
            max_groups=args.max_groups,
            preview_out=args.preview_out,
            preview_format=args.preview_format,
            only_file=args.only_file,
        )
        return
    else:
        ingest(
            data_dir=args.data_dir,
            persist_directory=args.persist_dir,
            model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            only_file=args.only_file,
            batch_size=DEFAULT_BATCH_SIZE,
        )


if __name__ == "__main__":
    main()
