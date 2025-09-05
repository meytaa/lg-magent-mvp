#!/usr/bin/env python3
"""
Test script to verify the new state management structure works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from lg_magent_mvp.app import AgentState
from lg_magent_mvp.state_manager import StateManager, write_context, write_trace, read_context, read_trace
from lg_magent_mvp.nodes import BaseNode
from lg_magent_mvp.nodes.plan import PlanNode
from lg_magent_mvp.nodes.summary import SummarizeDocNode
from lg_magent_mvp.nodes.approval import ApprovalNode


def test_state_manager_basic():
    """Test basic StateManager functionality."""
    print("=== Testing StateManager Basic Functionality ===")
    
    # Create a test state
    state: AgentState = {
        "question": "What are the key findings?",
        "doc_path": "test.pdf"
    }
    
    # Test initialization
    StateManager.initialize_state_data(state)
    assert "context_data" in state
    assert "trace_data" in state
    print("✓ State initialization works")
    
    # Test writing context data
    test_data = {"result": "test_result", "count": 5}
    StateManager.write_context_data(state, "test_node", test_data)
    assert state["context_data"]["test_node"] == test_data
    print("✓ Context data writing works")
    
    # Test reading context data
    read_data = StateManager.read_context_data(state, "test_node")
    assert read_data == test_data
    print("✓ Context data reading works")
    
    # Test writing trace data
    trace_data = {"detailed": "trace_info", "metadata": "extra"}
    StateManager.write_trace_data(state, "test_node", trace_data, {"source": "test"})
    assert "test_node" in state["trace_data"]
    assert state["trace_data"]["test_node"]["data"] == trace_data
    assert state["trace_data"]["test_node"]["metadata"]["source"] == "test"
    print("✓ Trace data writing works")
    
    # Test reading trace data
    read_trace_data = StateManager.read_trace_data(state, "test_node")
    assert read_trace_data["data"] == trace_data
    print("✓ Trace data reading works")
    
    print("✓ All StateManager basic tests passed!\n")


def test_convenience_functions():
    """Test convenience functions."""
    print("=== Testing Convenience Functions ===")
    
    state: AgentState = {"question": "Test question"}
    
    # Test convenience functions
    write_context(state, "conv_test", {"data": "context"})
    write_trace(state, "conv_test", {"data": "trace"}, {"meta": "data"})
    
    context_result = read_context(state, "conv_test")
    trace_result = read_trace(state, "conv_test")
    
    assert context_result["data"] == "context"
    assert trace_result["data"]["data"] == "trace"
    assert trace_result["metadata"]["meta"] == "data"
    
    print("✓ Convenience functions work correctly!\n")


def test_base_node_integration():
    """Test BaseNode integration with new state management."""
    print("=== Testing BaseNode Integration ===")
    
    class TestNode(BaseNode):
        def __init__(self):
            super().__init__("test_node", "A test node for verification")
        
        def execute(self, state: AgentState) -> AgentState:
            # Test the new helper methods
            self._write_context_data(state, {"processed": True, "items": 3})
            self._write_trace_data(state, {"details": "processing complete"}, {"method": "test"})
            return state
    
    # Create test node and state
    test_node = TestNode()
    state: AgentState = {"question": "Test"}
    
    # Execute the node
    result_state = test_node(state)
    
    # Verify context data was written
    context_data = result_state["context_data"]["test_node"]
    assert context_data["processed"] == True
    assert context_data["items"] == 3
    print("✓ BaseNode context data integration works")
    
    # Verify trace data was written
    trace_data = result_state["trace_data"]["test_node"]
    assert trace_data["data"]["details"] == "processing complete"
    assert trace_data["metadata"]["method"] == "test"
    print("✓ BaseNode trace data integration works")
    
    # Verify legacy notes still work
    assert "notes" in result_state
    assert any("Starting test_node" in note for note in result_state["notes"])
    print("✓ Legacy notes compatibility maintained")
    
    print("✓ BaseNode integration tests passed!\n")


def test_legacy_migration():
    """Test legacy state migration."""
    print("=== Testing Legacy State Migration ===")
    
    # Create a state with legacy data
    legacy_state: AgentState = {
        "question": "Test question",
        "doc_summary": {"pages": 10, "tables": 5},
        "findings": [{"type": "result", "data": "finding1"}],
        "notes": ["Starting planner", "keyword_search: found results"],
        "evidence": [{"type": "search", "data": "evidence1"}]
    }
    
    # Run migration
    StateManager.migrate_legacy_state(legacy_state)
    
    # Verify context_data was populated
    assert "context_data" in legacy_state
    assert "summarize" in legacy_state["context_data"]
    assert legacy_state["context_data"]["summarize"]["pages"] == 10
    print("✓ Legacy context data migration works")
    
    # Verify trace_data was populated
    assert "trace_data" in legacy_state
    assert "summarize" in legacy_state["trace_data"]
    assert legacy_state["trace_data"]["summarize"]["metadata"]["migrated_from"] == "doc_summary"
    print("✓ Legacy trace data migration works")
    
    print("✓ Legacy migration tests passed!\n")


def test_node_execution_with_state_management():
    """Test actual node execution with new state management."""
    print("=== Testing Node Execution with State Management ===")
    
    # Test ApprovalNode (simpler node)
    approval_node = ApprovalNode()
    state: AgentState = {"question": "Test"}
    
    # Execute approval node
    result_state = approval_node(state)
    
    # Verify new state structure was used
    assert "context_data" in result_state
    assert "approval" in result_state["context_data"]
    approval_context = result_state["context_data"]["approval"]
    assert approval_context["mode"] == "auto"
    assert approval_context["approved"] == True
    print("✓ ApprovalNode uses new state management")
    
    # Verify trace data
    assert "approval" in result_state["trace_data"]
    approval_trace = result_state["trace_data"]["approval"]
    assert approval_trace["metadata"]["auto_approved"] == True
    print("✓ ApprovalNode trace data works")
    
    print("✓ Node execution tests passed!\n")


def main():
    """Run all tests."""
    print("Testing New State Management Structure\n")
    
    try:
        test_state_manager_basic()
        test_convenience_functions()
        test_base_node_integration()
        test_legacy_migration()
        test_node_execution_with_state_management()
        
        print("🎉 All tests passed! The new state management structure is working correctly.")
        print("\nSummary of what was tested:")
        print("- Basic StateManager functionality (read/write context and trace data)")
        print("- Convenience functions for easier access")
        print("- BaseNode integration with new helper methods")
        print("- Legacy state migration for backward compatibility")
        print("- Actual node execution using the new structure")
        print("\nThe state now has:")
        print("- context_data: Dict[node_name, relevant_data] for immediately useful data")
        print("- trace_data: Dict[node_name, metadata_and_data] for detailed tracing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
