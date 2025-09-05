#!/usr/bin/env python3
"""
Example script showing how to use sentence transformers for embeddings.

This demonstrates:
1. How to configure sentence transformers via environment variables
2. How to use different embedding models
3. How to compare OpenAI vs Sentence Transformers embeddings

Usage:
    # Use sentence transformers with all-MiniLM-L6-v2 (default)
    EMBED_TYPE=sentence_transformers EMBED_MODEL=all-MiniLM-L6-v2 python examples/sentence_transformers_example.py
    
    # Use sentence transformers with a different model
    EMBED_TYPE=sentence_transformers EMBED_MODEL=all-mpnet-base-v2 python examples/sentence_transformers_example.py
    
    # Use OpenAI embeddings (default)
    EMBED_TYPE=openai EMBED_MODEL=text-embedding-3-small python examples/sentence_transformers_example.py
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import lg_magent_mvp
sys.path.insert(0, str(Path(__file__).parent.parent))

from lg_magent_mvp.models import get_embedder
from lg_magent_mvp.config import load_settings


def main():
    """Demonstrate different embedding options."""
    
    print("=== Embedding Configuration Example ===\n")
    
    # Load current settings
    settings = load_settings()
    print(f"Current settings:")
    print(f"  EMBED_MODEL: {settings.embed_model}")
    print(f"  EMBED_TYPE: {settings.embed_type}")
    print()
    
    # Test text
    test_texts = [
        "The patient has chronic back pain.",
        "Patient reports severe headaches lasting 3 days.",
        "Blood pressure is elevated at 150/90 mmHg.",
        "No significant medical history reported."
    ]
    
    try:
        # Get embedder based on current configuration
        embedder = get_embedder()
        print(f"Using embedder: {type(embedder).__name__}")
        
        if hasattr(embedder, 'model_name'):
            print(f"Model: {embedder.model_name}")
        elif hasattr(embedder, 'model'):
            print(f"Model: {embedder.model}")
        print()
        
        # Generate embeddings for test texts
        print("Generating embeddings...")
        embeddings = embedder.embed_documents(test_texts)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print()
        
        # Show first few dimensions of first embedding
        print("First embedding (first 10 dimensions):")
        print([round(x, 4) for x in embeddings[0][:10]])
        print()
        
        # Test query embedding
        query = "patient back pain"
        query_embedding = embedder.embed_query(query)
        print(f"Query embedding dimension: {len(query_embedding)}")
        print("Query embedding (first 10 dimensions):")
        print([round(x, 4) for x in query_embedding[:10]])
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf using sentence transformers, make sure to install dependencies:")
        print("pip install -e \".[sentence-transformers]\"")
        print("# or: pip install langchain-huggingface sentence-transformers")
        return 1
    
    print("\n=== Configuration Examples ===")
    print("\nTo use different embedding models, set environment variables:")
    print("\n1. Sentence Transformers (local, no API key needed):")
    print("   export EMBED_TYPE=sentence_transformers")
    print("   export EMBED_MODEL=all-MiniLM-L6-v2")
    print("   # or: all-mpnet-base-v2, BAAI/bge-small-en-v1.5, etc.")
    
    print("\n2. OpenAI Embeddings (requires API key):")
    print("   export EMBED_TYPE=openai")
    print("   export EMBED_MODEL=text-embedding-3-small")
    print("   # or: text-embedding-3-large, text-embedding-ada-002")
    
    print("\n3. Auto-detection (based on model name):")
    print("   export EMBED_TYPE=auto")
    print("   export EMBED_MODEL=all-MiniLM-L6-v2  # Will use sentence transformers")
    print("   export EMBED_MODEL=text-embedding-3-small  # Will use OpenAI")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
