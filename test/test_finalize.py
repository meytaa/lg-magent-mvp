from types import SimpleNamespace


def test_finalize_meta(monkeypatch):
    # Stub LLM to avoid network
    from lg_magent_mvp import nodes
    from lg_magent_mvp.nodes import finalize as fin_mod

    class DummyLLM:
        def invoke(self, msgs):
            return SimpleNamespace(content="stub narrative")

    monkeypatch.setattr(fin_mod, "get_finalize_llm", lambda: DummyLLM())

    state = {
        "question": "Q",
        "doc_path": "data/MC15 Deines Chiropractic.pdf",
        "doc_summary": {"pages": 5},
        "tables": [],
        "figures": [],
    }
    out = fin_mod.finalize_node(state)
    rep = out["report"]
    assert rep["report_meta"]["file"].endswith("MC15 Deines Chiropractic.pdf")
    assert rep["report_meta"]["pages"] == 5
    mv = rep["report_meta"]["model_versions"]
    assert set(["router_model", "finalize_model", "vision_model", "embed_model"]) <= set(mv.keys())
    assert out["narrative"] == "stub narrative"

