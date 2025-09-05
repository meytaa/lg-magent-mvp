from __future__ import annotations

from functools import lru_cache
from typing import Optional, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None  # type: ignore

from .config import Settings, load_settings


@lru_cache(maxsize=None)
def _settings() -> Settings:
    # Cache settings so repeated calls reuse same config
    return load_settings()


@lru_cache(maxsize=16)
def get_router_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    s = _settings()
    return ChatOpenAI(model=model_name or s.router_model)


@lru_cache(maxsize=16)
def get_finalize_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    s = _settings()
    return ChatOpenAI(model=model_name or s.finalize_model)


@lru_cache(maxsize=16)
def get_vision_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    s = _settings()
    return ChatOpenAI(model=model_name or s.vision_model)


@lru_cache(maxsize=16)
def get_orchestrator_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    s = _settings()
    # Use router model for orchestrator, or can be separate config
    return ChatOpenAI(model=model_name or s.router_model)


@lru_cache(maxsize=4)
def get_embedder(model_name: Optional[str] = None, embedding_type: Optional[str] = None) -> Union[OpenAIEmbeddings, "HuggingFaceEmbeddings"]:
    """Get embedder instance. Supports OpenAI and Sentence Transformers.

    Args:
        model_name: Specific model name to use
        embedding_type: 'openai' or 'sentence_transformers'. If None, uses config setting.

    Returns:
        Embeddings instance
    """
    s = _settings()
    model = model_name or s.embed_model
    embed_type = embedding_type or s.embed_type

    # Auto-detect embedding type if not specified and not in config
    if embed_type == "auto":
        if model.startswith(('all-MiniLM', 'all-mpnet', 'sentence-transformers/', 'BAAI/', 'intfloat/')):
            embed_type = 'sentence_transformers'
        else:
            embed_type = 'openai'

    if embed_type == 'sentence_transformers':
        if HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "HuggingFaceEmbeddings not available. Install with: "
                "pip install -e \".[sentence-transformers]\" or "
                "pip install langchain-huggingface sentence-transformers"
            )
        # Remove sentence-transformers/ prefix if present
        if model.startswith('sentence-transformers/'):
            model = model[len('sentence-transformers/'):]
        return HuggingFaceEmbeddings(model_name=model)
    else:
        return OpenAIEmbeddings(model=model)
