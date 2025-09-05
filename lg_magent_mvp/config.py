from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Central app settings derived from env and defaults.

    Step 1: encodes agreed preferences (CLI, LangSmith tracing, JSON+narrative output).
    Step 2: pins provider/models used by nodes (router/finalize/vision/embeddings).
    """

    # Provider
    provider: str = "openai"

    # Models (can be overridden via env)
    router_model: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    finalize_model: str = os.getenv("FINALIZE_MODEL", "gpt-4o")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_type: str = os.getenv("EMBED_TYPE", "openai")  # 'openai' or 'sentence_transformers'

    # Features/preferences from Step 1
    output_format: str = os.getenv("OUTPUT_FORMAT", "json+narrative")
    approvals: str = os.getenv("APPROVALS", "auto")  # or "pause-before-finalize"

    # Tracing (LangSmith via LangChain v2 tracing env vars)
    tracing: str = os.getenv("TRACING", "langsmith")  # or "none"
    project: str = os.getenv("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "medical-audit"))

    # Storage / indices
    faiss_dir: str = os.getenv("FAISS_DIR", ".cache/faiss")

    # Vision analysis options
    vision_use_images: bool = os.getenv("VISION_USE_IMAGES", "false").lower() in {"1", "true", "yes"}

    # Routing / control flow
    router_mode: str = os.getenv("ROUTER_MODE", "rule").lower()  # rule | llm | hybrid
    max_hops: int = int(os.getenv("MAX_HOPS", "12"))

    # Persistence / approvals
    use_memory: bool = os.getenv("USE_MEMORY", "true").lower() in {"1", "true", "yes"}
    checkpoint_db: str = os.getenv("CHECKPOINT_DB", ".cache/graph_state.sqlite")

    # Logging / observability
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    json_logs: bool = os.getenv("JSON_LOGS", "false").lower() in {"1", "true", "yes"}


def load_settings() -> Settings:
    return Settings()


def ensure_env(settings: Settings) -> None:
    """Validate required environment for chosen provider and tracing.

    - Requires OPENAI_API_KEY for provider=openai
    - For LangSmith, expects LANGCHAIN_API_KEY (accepts LANGSMITH_API_KEY alias)
    """

    if settings.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider")

    # LangSmith keys (optional but recommended)
    if settings.tracing == "langsmith":
        # Support either LANGCHAIN_API_KEY or legacy alias LANGSMITH_API_KEY
        if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
        # Map endpoint alias if provided
        if not os.getenv("LANGCHAIN_ENDPOINT") and os.getenv("LANGSMITH_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]


def apply_tracing_env(settings: Settings) -> None:
    """Enable LangSmith tracing through LangChain if requested."""
    if settings.tracing == "langsmith":
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.project)
        # Endpoint can be configured if self-hosted; default SaaS otherwise
        if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]
    else:
        # Ensure tracing is off if explicitly disabled
        os.environ.pop("LANGCHAIN_TRACING_V2", None)


def load_env_from_dotenv() -> None:
    """Load environment variables from a .env file if present."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        # Graceful no-op if python-dotenv isn't installed
        pass
