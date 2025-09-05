"""
State management utilities for context_data and trace_data.

This module provides utilities for managing the new structured state format where:
- context_data: Dict[node_name, relevant_data] - immediately useful data
- trace_data: Dict[node_name, metadata_and_data] - includes metadata and trace info
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import time
from datetime import datetime

if TYPE_CHECKING:
    from .app import AgentState


class StateManager:
    """Utility class for managing context_data and trace_data in AgentState."""
    
    @staticmethod
    def initialize_state_data(state: "AgentState") -> None:
        """Initialize context_data and trace_data if they don't exist."""
        if "context_data" not in state:
            state["context_data"] = {}
        if "trace_data" not in state:
            state["trace_data"] = {}
    
    @staticmethod
    def write_context_data(state: "AgentState", node_name: str, data: Any) -> None:
        """
        Write data to context_data for a specific node.
        
        Args:
            state: The agent state
            node_name: Name of the node writing the data
            data: The relevant data to store (should be immediately useful)
        """
        StateManager.initialize_state_data(state)
        state["context_data"][node_name] = data
    
    @staticmethod
    def write_trace_data(state: "AgentState", node_name: str, data: Any, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Write data to trace_data for a specific node with metadata.
        
        Args:
            state: The agent state
            node_name: Name of the node writing the data
            data: The data to store (can include metadata and trace info)
            metadata: Optional additional metadata to include
        """
        StateManager.initialize_state_data(state)
        
        trace_entry = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name,
        }
        
        if metadata:
            trace_entry["metadata"] = metadata
            
        state["trace_data"][node_name] = trace_entry
    
    @staticmethod
    def read_context_data(state: "AgentState", node_name: str) -> Any:
        """
        Read context data for a specific node.
        
        Args:
            state: The agent state
            node_name: Name of the node to read data for
            
        Returns:
            The context data for the node, or None if not found
        """
        StateManager.initialize_state_data(state)
        return state["context_data"].get(node_name)
    
    @staticmethod
    def read_trace_data(state: "AgentState", node_name: str) -> Any:
        """
        Read trace data for a specific node.
        
        Args:
            state: The agent state
            node_name: Name of the node to read data for
            
        Returns:
            The trace data for the node, or None if not found
        """
        StateManager.initialize_state_data(state)
        return state["trace_data"].get(node_name)
    
    @staticmethod
    def get_all_context_nodes(state: "AgentState") -> list:
        """Get list of all node names that have context data."""
        StateManager.initialize_state_data(state)
        return list(state["context_data"].keys())
    
    @staticmethod
    def get_all_trace_nodes(state: "AgentState") -> list:
        """Get list of all node names that have trace data."""
        StateManager.initialize_state_data(state)
        return list(state["trace_data"].keys())
    
    @staticmethod
    def migrate_legacy_state(state: "AgentState") -> None:
        """
        Migrate existing state data to the new context_data/trace_data format.
        This helps with backward compatibility.
        """
        StateManager.initialize_state_data(state)
        
        # Migrate existing data based on current state structure
        migrations = {
            # Map existing state fields to context data
            "doc_summary": "summarize",
            "findings": "search_results", 
            "tables": "extract_tables",
            "figures": "analyze_figures",
            "plan": "planner",
            "narrative": "finalize",
            "report": "finalize"
        }
        
        for state_key, node_name in migrations.items():
            if state_key in state and state[state_key] is not None:
                # Only migrate if we don't already have context data for this node
                if node_name not in state["context_data"]:
                    StateManager.write_context_data(state, node_name, state[state_key])
                    
                    # Also add to trace data with migration metadata
                    StateManager.write_trace_data(
                        state, 
                        node_name, 
                        state[state_key],
                        metadata={"migrated_from": state_key, "migration_time": datetime.now().isoformat()}
                    )
        
        # Migrate notes and evidence to trace data if they exist
        if "notes" in state and state["notes"]:
            for i, note in enumerate(state["notes"]):
                # Try to extract node name from note format "Starting node_name" or "node_name: message"
                node_name = "general"  # default
                if note.startswith("Starting "):
                    node_name = note.replace("Starting ", "").split()[0]
                elif ": " in note:
                    potential_node = note.split(": ")[0]
                    if potential_node.replace("_", "").isalnum():  # basic validation
                        node_name = potential_node
                
                # Add to trace data
                trace_key = f"{node_name}_note_{i}"
                if trace_key not in state["trace_data"]:
                    StateManager.write_trace_data(
                        state,
                        trace_key,
                        {"note": note, "type": "legacy_note"},
                        metadata={"migrated_from": "notes", "note_index": i}
                    )
        
        if "evidence" in state and state["evidence"]:
            for i, evidence in enumerate(state["evidence"]):
                evidence_type = evidence.get("type", "unknown")
                trace_key = f"evidence_{evidence_type}_{i}"
                if trace_key not in state["trace_data"]:
                    StateManager.write_trace_data(
                        state,
                        trace_key,
                        evidence,
                        metadata={"migrated_from": "evidence", "evidence_index": i}
                    )


# Convenience functions for easier access
def write_context(state: "AgentState", node_name: str, data: Any) -> None:
    """Convenience function to write context data."""
    StateManager.write_context_data(state, node_name, data)


def write_trace(state: "AgentState", node_name: str, data: Any, 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to write trace data."""
    StateManager.write_trace_data(state, node_name, data, metadata)


def read_context(state: "AgentState", node_name: str) -> Any:
    """Convenience function to read context data."""
    return StateManager.read_context_data(state, node_name)


def read_trace(state: "AgentState", node_name: str) -> Any:
    """Convenience function to read trace data."""
    return StateManager.read_trace_data(state, node_name)
