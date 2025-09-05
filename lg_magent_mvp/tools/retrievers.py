from __future__ import annotations

import re
from typing import Dict, List, Tuple, Iterable, Optional

from ..models import get_embedder
from ..pdf import extract_text_pages
from ..indexing import ensure_faiss_index, make_snippet

try:
    from langchain_community.vectorstores import FAISS
except Exception:  # pragma: no cover - import error handled at runtime
    FAISS = None  # type: ignore


DEFAULT_DOC_PATH = "data/MC15 Deines Chiropractic.pdf"


# -------- Keyword search helpers --------

def keyword_matches_in_page(text: str, query: str) -> Iterable[Tuple[int, int]]:
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for m in pattern.finditer(text or ""):
        yield m.start(), m.end()


def keyword_search(query: str, doc_path: Optional[str] = None, max_hits_per_page: int = 5) -> List[Dict]:
    """Search for keyword matches across the whole document text and return
    a snippet (chunk) that may span page boundaries.

    - Pages are concatenated with separators and searched as a single string.
    - For each match, we determine the enclosing page (based on match start).
    - The returned snippet is taken from the concatenated text, allowing it to
      include neighboring content that may lie on an adjacent page.
    - We still limit the number of hits per page via ``max_hits_per_page``.
    """
    if not query:
        return []

    doc = doc_path or DEFAULT_DOC_PATH
    pages = extract_text_pages(doc)
    hits: List[Dict] = []

    # Build concatenated text and track page start offsets within it
    sep = "\n\n"
    parts: List[str] = []
    offsets: List[Tuple[int, int, int]] = []  # (page_no, base_offset, text_len)
    base = 0
    for p in pages:
        page_no = p["page"]
        text = p.get("text") or ""
        parts.append(text)
        offsets.append((page_no, base, len(text)))
        base += len(text) + len(sep)
    concat_text = sep.join(parts)

    # Helper: find page index for a global position
    def _page_index_for(pos: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            _, base_off, tlen = offsets[mid]
            if base_off <= pos < base_off + tlen:
                return mid
            if pos < base_off:
                hi = mid - 1
            else:
                lo = mid + 1
        # If position falls into a separator, attribute to previous page when possible
        return max(0, min(lo - 1, len(offsets) - 1))

    # Track per-page hit counts
    page_counts: Dict[int, int] = {}

    for g_start, g_end in keyword_matches_in_page(concat_text, query):
        pi = _page_index_for(g_start)
        page_no, base_off, tlen = offsets[pi]
        page_counts.setdefault(page_no, 0)
        if page_counts[page_no] >= max_hits_per_page:
            continue

        # Compute page-local span for the match
        l_start = max(0, g_start - base_off)
        l_end = min(tlen, g_end - base_off)

        # Create a snippet from the concatenated text to allow cross-page context
        snippet = make_snippet(concat_text, g_start, g_end)

        hits.append({
            "doc": doc,
            "page": page_no,
            "span": [l_start, l_end],
            "snippet": snippet,
        })
        page_counts[page_no] += 1

    return hits


# -------- Semantic search via FAISS --------


def semantic_search(query: str, doc_path: Optional[str] = None, k: int = 5) -> List[Dict]:
    """Search for semantically similar chunks using FAISS vector index."""
    doc = doc_path or DEFAULT_DOC_PATH
    idx_dir = ensure_faiss_index(doc)
    embedder = get_embedder()
    if FAISS is None:
        raise RuntimeError("FAISS is not available. Ensure langchain-community and faiss-cpu are installed.")
    store = FAISS.load_local(idx_dir, embeddings=embedder, allow_dangerous_deserialization=True)
    results = store.similarity_search_with_score(query, k=k)
    hits: List[Dict] = []
    for doc_chunk, score in results:
        md = doc_chunk.metadata or {}
        hits.append({
            "doc": md.get("doc", doc),
            "page": int(md.get("page", 1)),
            "span": md.get("span", [0, 0]),
            "snippet": md.get("snippet", doc_chunk.page_content[:300]),
            "score": float(score),
            "chunk_id": md.get("chunk_id", ""),
        })
    return hits


# Backward-compatible names used by nodes
def keyword_retriever(query: str, *, doc_path: Optional[str] = None) -> List[Dict]:
    return keyword_search(query, doc_path=doc_path)


def semantic_retriever(query: str, *, doc_path: Optional[str] = None, k: int = 5) -> List[Dict]:
    return semantic_search(query, doc_path=doc_path, k=k)


def table_extractor(documents: list) -> list:
    """Extract tables from documents. Currently not implemented."""
    _ = documents  # Suppress unused parameter warning
    return []


def figure_analyzer(images: list) -> list:
    """Analyze figures in images. Currently not implemented."""
    _ = images  # Suppress unused parameter warning
    return []
