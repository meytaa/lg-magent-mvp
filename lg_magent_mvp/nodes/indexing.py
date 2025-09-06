from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Any, Optional
import json
import hashlib
import os
from pathlib import Path
import numpy as np
from openai import OpenAI

from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better semantic search."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', start + chunk_size - 100, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def create_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Create embeddings for text chunks using OpenAI text-embedding-3-small."""
    if not texts:
        return np.array([])
    
    print(f"  - Creating embeddings for {len(texts)} text chunks...")
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = np.array([item.embedding for item in response.data])
        print(f"  - Created embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"  - Error creating embeddings: {e}")
        return np.array([])


def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """Build FAISS index from embeddings."""
    if not FAISS_AVAILABLE or embeddings.size == 0:
        return None
    
    print(f"  - Building FAISS index with {embeddings.shape[0]} vectors...")
    
    try:
        # Use IndexFlatIP for cosine similarity (after normalization)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        print(f"  - FAISS index built successfully with {index.ntotal} vectors")
        return index
        
    except Exception as e:
        print(f"  - Error building FAISS index: {e}")
        return None


def extract_text_from_summary(summary_data: Dict) -> List[Dict[str, Any]]:
    """Extract text content from summary data for indexing."""
    text_chunks = []
    
    # Get document content from summary
    document_content = summary_data.get("document_content", [])
    
    for page_data in document_content:
        page_num = page_data.get("page", 1)
        section = page_data.get("section", "Unknown")
        
        # Extract text from page content
        page_content = page_data.get("page_content", [])
        page_texts = []
        
        for content_item in page_content:
            if content_item.get("type") == "text":
                text = content_item.get("content", "")
                if text and text.strip():
                    page_texts.append(text.strip())
        
        # Also get integrated text if available
        integrated_text = page_data.get("integrated_text", "")
        if integrated_text and integrated_text.strip():
            page_texts.append(integrated_text.strip())
        
        # Combine all text from this page
        if page_texts:
            full_page_text = " ".join(page_texts)
            
            # Chunk the page text
            chunks = chunk_text(full_page_text)
            
            for i, chunk in enumerate(chunks):
                text_chunks.append({
                    "text": chunk,
                    "page": page_num,
                    "section": section,
                    "chunk_id": f"page_{page_num}_chunk_{i+1}",
                    "metadata": {
                        "page": page_num,
                        "section": section,
                        "chunk_index": i,
                        "total_chunks_on_page": len(chunks)
                    }
                })
    
    return text_chunks


def create_text_index(summary_data: Dict, output_dir: str) -> Dict[str, Any]:
    """Create FAISS text index from summary data."""
    if not FAISS_AVAILABLE:
        print("⚠️ FAISS not available - skipping text indexing")
        return {"indexed": False, "reason": "FAISS not available"}
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("🔄 Creating text index from summary data...")
    
    # Extract text chunks from summary
    text_chunks = extract_text_from_summary(summary_data)
    
    if not text_chunks:
        print("⚠️ No text content found in summary data")
        return {"indexed": False, "reason": "No text content"}
    
    print(f"  - Extracted {len(text_chunks)} text chunks from summary")
    
    # Create embeddings
    texts = [chunk["text"] for chunk in text_chunks]
    embeddings = create_embeddings(texts, client)
    
    if embeddings.size == 0:
        print("⚠️ Failed to create embeddings")
        return {"indexed": False, "reason": "Embedding creation failed"}
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    if index is None:
        print("⚠️ Failed to build FAISS index")
        return {"indexed": False, "reason": "FAISS index creation failed"}
    
    # Save index and metadata
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = output_path / "text_index.faiss"
    faiss.write_index(index, str(index_path))
    
    # Save text chunks metadata
    metadata_path = output_path / "text_chunks.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(text_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved FAISS index to: {index_path}")
    print(f"💾 Saved metadata to: {metadata_path}")
    
    return {
        "indexed": True,
        "total_chunks": len(text_chunks),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embedding_dimension": embeddings.shape[1]
    }


class IndexingNode(BaseNode):
    """Node for creating FAISS text index from summary data."""

    def __init__(self):
        super().__init__(
            "indexing",
            description="Creates FAISS text index from summary data for semantic search using OpenAI embeddings."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        doc_path = self._get_doc_path(state)
        if not doc_path:
            return state

        # Check cache first
        cache_key = self._get_cache_key(doc_path)
        cached_result = self._load_from_cache(cache_key, "indexing")
        
        if cached_result:
            print("✅ Using cached text index")
            self._write_context_data(state, cached_result)
            self._add_note(state, f"Used cached text index with {cached_result.get('total_chunks', 0)} chunks")
            return state

        # Get summary data from state (should be available from summary node)
        summary_data = state.get("doc_summary_raw")  # Raw summary data with full content
        
        if not summary_data:
            print("⚠️ No summary data available - run summary node first")
            self._add_note(state, "Indexing failed: No summary data available")
            return state

        # Create output directory
        output_dir = f"output/indexing/{Path(doc_path).stem}"
        
        try:
            print("🔄 Creating text index from summary data (no cache found)")
            
            # Create text index
            result = create_text_index(summary_data, output_dir)
            
            # Cache the result
            self._save_to_cache(cache_key, result, "indexing")
            
            # Store in context_data
            self._write_context_data(state, result)
            
            if result.get("indexed"):
                self._add_note(state, f"Created text index with {result['total_chunks']} chunks")
            else:
                self._add_note(state, f"Indexing failed: {result.get('reason', 'Unknown error')}")
            
        except Exception as e:
            self._add_note(state, f"Indexing failed: {str(e)}")
            self._write_context_data(state, {"indexed": False, "error": str(e)})
        
        return state


# Create instance for backward compatibility
indexing_node = IndexingNode()
