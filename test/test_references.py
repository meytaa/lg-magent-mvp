#!/usr/bin/env python3

import json
from lg_magent_mvp.config import load_env_from_dotenv, load_settings, apply_tracing_env, ensure_env
from lg_magent_mvp.app import compile_graph_with_settings

def test_references():
    """Test the reference tracking in the system."""
    
    # Setup
    load_env_from_dotenv()
    settings = load_settings()
    apply_tracing_env(settings)
    ensure_env(settings)
    
    graph = compile_graph_with_settings(settings)
    
    # Run the system
    state = {
        "question": "What medical conditions are shown in the spine diagrams?",
        "doc_path": "data/MM155 Deines Chiropractic-1.pdf",
    }
    
    result = graph.invoke(state, config={"configurable": {"thread_id": "test_references"}})
    
    print("=== FINAL ANSWER ===")
    print(result.get("answer", "No answer found"))
    print()
    
    print("=== ORCHESTRATOR CHAT HISTORY ===")
    chat_history = result.get("orchestrator_chat_history", [])
    for i, entry in enumerate(chat_history):
        print(f"Entry {i+1}:")
        print(f"Role: {entry.get('role', 'unknown')}")
        print(f"Content: {entry.get('content', '')[:500]}...")  # Truncate for readability
        print("---")
    
    print(f"\n=== AGENT RESULTS ===")
    agent_results = result.get("agent_results", [])
    for i, agent_result in enumerate(agent_results):
        print(f"Agent {i+1}: {agent_result.get('agent', 'unknown')}")
        print(f"Success: {agent_result.get('success', False)}")
        data = agent_result.get('data', {})
        if 'analyses' in data:
            analyses = data['analyses']
            print(f"Figure analyses: {len(analyses)}")
            for j, analysis in enumerate(analyses[:2]):  # Show first 2
                print(f"  Analysis {j+1}: {analysis.get('figure_id', 'unknown')} - {analysis.get('analysis', '')[:100]}...")
        print("---")

if __name__ == "__main__":
    test_references()
