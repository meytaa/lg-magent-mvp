import os


def test_approval_routing_and_node(monkeypatch):
    # Configure approvals to pause
    monkeypatch.setenv("APPROVALS", "pause-before-finalize")

    from lg_magent_mvp.app import rule_router
    from lg_magent_mvp.nodes.approval import approval_node

    # State that has finished plan and has evidence
    state = {
        "doc_summary": {"pages": 1, "counts": {}},
        "plan": [],
        "cursor": 0,
        "evidence": [{"type": "semantic_hits", "data": [1]}],
    }

    # Router should ask for approval
    assert rule_router(state) == "approval"

    # Approval node marks awaiting
    state = approval_node(state)
    assert state.get("awaiting_approval") is True

    # Router should end (pause) until approved
    from langgraph.graph import END
    assert rule_router(state) == END

    # Approve and now router should go to finalize
    state["approved"] = True
    assert rule_router(state) == "finalize"

