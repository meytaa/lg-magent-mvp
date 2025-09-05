from lg_magent_mvp.app import rule_router, AgentState, _has_any_evidence


def test_rule_router_paths():
    # No doc summary yet -> summarize
    s: AgentState = {"question": "Q"}
    assert rule_router(s) == "summarize"

    # Has summary but no plan -> planner
    s["doc_summary"] = {"pages": 1, "counts": {}}
    assert rule_router(s) == "planner"

    # Has plan -> first step
    s["plan"] = ["keyword_search", "semantic_search"]
    s["cursor"] = 0
    assert rule_router(s) == "keyword_search"

    # After finishing plan with evidence -> finalize
    s["cursor"] = 3
    s["evidence"] = [{"type": "keyword_hits", "data": [1]}]
    assert _has_any_evidence(s) is True
    assert rule_router(s) == "finalize"

    # Done -> END
    s["done"] = True
    from langgraph.graph import END
    assert rule_router(s) == END
