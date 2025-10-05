from collections import Counter
import os
import sys

# Ensure workspace root is on sys.path when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_pipeline import load_vector_store, DEFAULT_CHROMA_DIR

def main() -> None:
    vs = load_vector_store(persist_directory=DEFAULT_CHROMA_DIR)
    col = getattr(vs, "_collection", None)
    if col is None:
        print("No Chroma collection found at:", DEFAULT_CHROMA_DIR)
        return

    src_counter: Counter[str] = Counter()
    pid_counter: Counter[str] = Counter()
    total = 0
    offset = 0
    page = 1000
    while True:
        try:
            res = col.get(include=["metadatas"], limit=page, offset=offset)
        except Exception as e:
            print("Error reading collection:", e)
            break
        if not isinstance(res, dict):
            break
        ids = res.get("ids", [])
        if not ids:
            break
        metadatas = res.get("metadatas", [])
        for md in metadatas:
            md = md or {}
            src = md.get("source", "unknown")
            pid = md.get("paper_id", "unknown")
            src_counter[src] += 1
            pid_counter[pid] += 1
            total += 1
        if len(ids) < page:
            break
        offset += page

    print(f"Total chunks: {total}")
    print("\nBy source (top 50):")
    for src, c in src_counter.most_common(50):
        print(f"{c}\t{src}")
    print("\nBy paper_id (top 50):")
    for pid, c in pid_counter.most_common(50):
        print(f"{c}\t{pid}")

if __name__ == "__main__":
    main()
