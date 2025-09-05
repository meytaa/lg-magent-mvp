import os
import time
import pytest


@pytest.mark.e2e
def test_e2e_run_report(monkeypatch):
    # Only run when explicitly enabled and keys are present
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY")
    if os.getenv("E2E") not in {"1", "true", "TRUE"}:
        pytest.skip("Set E2E=1 to enable end-to-end test")

    # Prefer rule routing and auto approvals for deterministic run
    monkeypatch.setenv("ROUTER_MODE", "rule")
    monkeypatch.setenv("APPROVALS", "auto")
    monkeypatch.setenv("USE_MEMORY", "false")

    from dotenv import load_dotenv  # type: ignore
    load_dotenv()

    from lg_magent_mvp.config import load_settings, apply_tracing_env, ensure_env
    from lg_magent_mvp.tools.retrievers import ensure_faiss_index
    from lg_magent_mvp.app import compile_graph_with_settings

    settings = load_settings()
    apply_tracing_env(settings)
    ensure_env(settings)

    # Warm FAISS index for speed; first run may take a while
    ensure_faiss_index("data/MC15 Deines Chiropractic.pdf")

    graph = compile_graph_with_settings(settings)

    t0 = time.time()
    out = graph.invoke({
        "question": "Audit this medical document for completeness, coding, and quality.",
        "doc_path": "data/MC15 Deines Chiropractic.pdf",
    }, config={"configurable": {"thread_id": "e2e-test"}})
    elapsed = time.time() - t0

    assert out.get("narrative"), "Should produce an executive narrative"
    rep = out.get("report") or {}
    assert rep, "Should produce a structured report"
    meta = rep.get("report_meta", {})
    assert meta.get("pages", 0) > 0
    models = meta.get("model_versions", {})
    for k in ["router_model", "finalize_model", "embed_model"]:
        assert k in models
    # Evidence should contain at least one category
    ev = out.get("evidence", [])
    assert any(e.get("type") in {"keyword_hits", "semantic_hits", "tables", "figures"} for e in ev)
    # Should have finalized
    assert out.get("done") is True

