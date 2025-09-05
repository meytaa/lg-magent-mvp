from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, List, Tuple

from .config import load_settings
from .models import get_embedder
from .pdf import extract_text_pages

try:
    from langchain_community.vectorstores import FAISS
except Exception:  # pragma: no cover - import error handled at runtime
    FAISS = None  # type: ignore

if TYPE_CHECKING:
    from .app import AgentState

# Chunking configuration (can be overridden via env if needed)
CHUNK_SIZE = int(os.environ.get("FAISS_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("FAISS_CHUNK_OVERLAP", "150"))


@dataclass
class Chunk:
    page: int
    start: int
    end: int
    text: str
    id: str


def make_snippet(text: str, start: int, end: int, window: int = 80) -> str:
    """Create a snippet around a text match for display purposes."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = text[left:start]
    match = text[start:end]
    suffix = text[end:right]
    snippet = (prefix + match + suffix).strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet[:300]


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[Tuple[int, int]]:
    """Split text into overlapping chunks."""
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + size)
        spans.append((i, j))
        if j == n:
            break
        i = max(i + size - overlap, j)  # ensure progress
    return spans


def build_chunks(pages: List[Dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Build chunks from document pages."""
    chunks: List[Chunk] = []
    for p in pages:
        page_no, text = p["page"], p["text"] or ""
        spans = chunk_text(text, size=size, overlap=overlap)
        for idx, (s, e) in enumerate(spans, start=1):
            chunk_text_value = text[s:e]
            cid = f"{page_no}-{idx}"
            chunks.append(Chunk(page=page_no, start=s, end=e, text=chunk_text_value, id=cid))
    return chunks


def _doc_key(doc_path: str, *, salt: str = "") -> str:
    """Generate a unique key for a document based on path and metadata."""
    h = hashlib.sha256()
    h.update(doc_path.encode())
    if salt:
        h.update(salt.encode())
    try:
        # include file size and mtime for invalidation
        st = os.stat(doc_path)
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    except Exception:
        pass
    return h.hexdigest()[:16]


def _index_dir(base_dir: str, key: str) -> str:
    """Get the directory path for an index."""
    return os.path.join(base_dir, key)


def ensure_faiss_index(doc_path: str) -> str:
    """Ensure a FAISS index exists for the given document, creating it if necessary."""
    settings = load_settings()
    os.makedirs(settings.faiss_dir, exist_ok=True)
    embedder = get_embedder()
    salt = f"embed={getattr(embedder, 'model', 'openai')}|cs={CHUNK_SIZE}|co={CHUNK_OVERLAP}"
    key = _doc_key(doc_path, salt=salt)
    idx_dir = _index_dir(settings.faiss_dir, key)
    meta_path = os.path.join(idx_dir, "meta.json")

    if os.path.exists(idx_dir) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if (
                meta.get("embed_model") == getattr(embedder, "model", "openai")
                and meta.get("chunk_size") == CHUNK_SIZE
                and meta.get("chunk_overlap") == CHUNK_OVERLAP
            ):
                return idx_dir
        except Exception:
            # fall through to rebuild
            pass

    if FAISS is None:
        raise RuntimeError("FAISS is not available. Ensure langchain-community and faiss-cpu are installed.")

    pages = extract_text_pages(doc_path)
    chunks = build_chunks(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    texts = [c.text for c in chunks]
    metadatas = [{
        "doc": doc_path,
        "page": c.page,
        "span": [c.start, c.end],
        "chunk_id": c.id,
        "snippet": make_snippet(c.text, 0, min(len(c.text), 80)),
    } for c in chunks]

    vs = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
    vs.save_local(idx_dir)

    os.makedirs(idx_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({
            "doc": doc_path,
            "chunks": len(chunks),
            "embed_model": getattr(embedder, "model", "openai"),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "version": 1,
        }, f)
    return idx_dir


def index_document(state: "AgentState") -> "AgentState":
    """Build or load a FAISS vector index for the current document.

    - Chunks the document text
    - Embeds chunks using the configured embedder
    - Stores vectors in a FAISS index on disk

    Adds the following fields to state:
    - state["document_indexed"]: bool
    - state["faiss_index_dir"]: str (path to index directory)
    - state["index_meta"]: Dict[str, Any] (summary of the index)
    """

    doc_path = state.get("doc_path")
    if not doc_path:
        raise ValueError("No document path provided")

    idx_dir = ensure_faiss_index(doc_path)

    meta: Dict[str, Any] = {}
    meta_path = os.path.join(idx_dir, "meta.json")
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        # If meta is missing for any reason, fall back to minimal info
        meta = {"doc": doc_path}

    state["document_indexed"] = True
    state["faiss_index_dir"] = idx_dir
    state["index_meta"] = meta
    return state

