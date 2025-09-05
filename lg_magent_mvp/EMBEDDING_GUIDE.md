# Embedding and Indexing Guide

This guide explains the refactored embedding and indexing system in the LangGraph Medical Agent MVP.

## Architecture Changes

### Separation of Concerns

The codebase has been refactored to separate indexing and retrieval responsibilities:

- **`lg_magent_mvp/indexing.py`**: Contains all document indexing functionality
- **`lg_magent_mvp/tools/retrievers.py`**: Contains only search/retrieval functionality
- **`lg_magent_mvp/models.py`**: Enhanced with multiple embedding provider support

### Key Benefits

1. **Clear separation**: Indexing logic is isolated from search logic
2. **Flexibility**: Easy to swap embedding providers without changing search code
3. **Maintainability**: Each module has a single, well-defined responsibility

## Embedding Options

### 1. OpenAI Embeddings (Default)

Uses OpenAI's embedding API. Requires an API key but provides high-quality embeddings.

```bash
export EMBED_TYPE=openai
export EMBED_MODEL=text-embedding-3-small
export OPENAI_API_KEY=your_api_key_here
```

**Available Models:**
- `text-embedding-3-small` (1536 dimensions, cost-effective)
- `text-embedding-3-large` (3072 dimensions, highest quality)
- `text-embedding-ada-002` (1536 dimensions, legacy)

### 2. Sentence Transformers (Local)

Uses Hugging Face sentence transformers. Runs locally, no API key required.

```bash
export EMBED_TYPE=sentence_transformers
export EMBED_MODEL=all-MiniLM-L6-v2
```

**Popular Models:**
- `all-MiniLM-L6-v2` (384 dimensions, fast and lightweight)
- `all-mpnet-base-v2` (768 dimensions, better quality)
- `BAAI/bge-small-en-v1.5` (384 dimensions, optimized for retrieval)
- `intfloat/e5-small-v2` (384 dimensions, multilingual)

**Installation:**
```bash
# Option 1: Install with optional dependencies
pip install -e ".[sentence-transformers]"

# Option 2: Install manually
pip install langchain-huggingface sentence-transformers
```

### 3. Auto-Detection

Automatically chooses the embedding type based on the model name:

```bash
export EMBED_TYPE=auto
export EMBED_MODEL=all-MiniLM-L6-v2  # Will use sentence transformers
# or
export EMBED_MODEL=text-embedding-3-small  # Will use OpenAI
```

## Usage Examples

### Basic Configuration

```python
from lg_magent_mvp.models import get_embedder

# Use default configuration
embedder = get_embedder()

# Override model
embedder = get_embedder(model_name="all-mpnet-base-v2")

# Override type
embedder = get_embedder(embedding_type="sentence_transformers")
```

### Indexing Documents

```python
from lg_magent_mvp.indexing import ensure_faiss_index, index_document

# Create/load FAISS index for a document
index_dir = ensure_faiss_index("path/to/document.pdf")

# Use in LangGraph state (for nodes)
state = {"doc_path": "path/to/document.pdf"}
updated_state = index_document(state)
```

### Searching Documents

```python
from lg_magent_mvp.tools.retrievers import semantic_search, keyword_search

# Semantic search using embeddings
results = semantic_search("patient symptoms", doc_path="document.pdf", k=5)

# Keyword search
results = keyword_search("blood pressure", doc_path="document.pdf")
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_TYPE` | `openai` | Embedding provider: `openai`, `sentence_transformers`, or `auto` |
| `EMBED_MODEL` | `text-embedding-3-small` | Model name for the chosen provider |
| `FAISS_CHUNK_SIZE` | `800` | Text chunk size for indexing |
| `FAISS_CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `FAISS_DIR` | `.cache/faiss` | Directory for FAISS indices |

### Model Recommendations

**For Production (High Quality):**
- OpenAI: `text-embedding-3-large`
- Sentence Transformers: `all-mpnet-base-v2`

**For Development (Fast/Cheap):**
- OpenAI: `text-embedding-3-small`
- Sentence Transformers: `all-MiniLM-L6-v2`

**For Offline/Privacy:**
- Sentence Transformers: Any model (runs locally)

## Migration Guide

### From Old System

If you were using the old system where indexing was mixed with retrieval:

1. **No code changes needed** for basic usage - the public API remains the same
2. **Import changes**: If you were importing indexing functions directly:
   ```python
   # Old
   from lg_magent_mvp.tools.retrievers import ensure_faiss_index
   
   # New
   from lg_magent_mvp.indexing import ensure_faiss_index
   ```

### Testing the Setup

Run the example script to test your configuration:

```bash
python lg_magent_mvp/examples/sentence_transformers_example.py
```

## Troubleshooting

### Common Issues

1. **"HuggingFaceEmbeddings not available"**
   ```bash
   # Install with optional dependencies
   pip install -e ".[sentence-transformers]"

   # Or install manually
   pip install langchain-huggingface sentence-transformers
   ```

2. **"FAISS is not available"**
   ```bash
   pip install faiss-cpu
   # or for GPU support:
   pip install faiss-gpu
   ```

3. **OpenAI API errors**
   - Check your `OPENAI_API_KEY` environment variable
   - Verify you have sufficient API credits

4. **Memory issues with large models**
   - Use smaller models like `all-MiniLM-L6-v2`
   - Reduce `FAISS_CHUNK_SIZE` if needed

### Performance Tips

1. **Choose appropriate models**: Smaller models are faster but may be less accurate
2. **Tune chunk size**: Larger chunks capture more context but use more memory
3. **Cache embeddings**: The system automatically caches FAISS indices
4. **Use local models**: Sentence transformers avoid API latency

## Advanced Usage

### Custom Embedding Models

You can extend the system to support additional embedding providers by modifying `lg_magent_mvp/models.py`:

```python
def get_embedder(model_name=None, embedding_type=None):
    # Add your custom provider here
    if embedding_type == 'custom_provider':
        return YourCustomEmbeddings(model=model_name)
    # ... existing code
```

### Batch Processing

For processing multiple documents:

```python
from lg_magent_mvp.indexing import ensure_faiss_index

documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for doc in documents:
    index_dir = ensure_faiss_index(doc)
    print(f"Indexed {doc} -> {index_dir}")
```
