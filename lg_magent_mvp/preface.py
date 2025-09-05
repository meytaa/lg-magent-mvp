from typing import TYPE_CHECKING
from .nodes.summary import summarize_doc_node
from .nodes.indexing import indexing_node
from .nodes import BaseNode

if TYPE_CHECKING:
    from .app import AgentState


class PrefaceNode(BaseNode):
    """Preface node that orchestrates document summarization and indexing."""

    def __init__(self):
        super().__init__(
            "preface",
            description="Orchestrates document summarization and hybrid content indexing as initial processing steps."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        """Execute preface: run summary and indexing nodes in sequence."""

        # Step 1: Document summarization
        self._add_note(state, "Starting document summarization...")
        state = summarize_doc_node(state)

        # Step 2: Hybrid content indexing
        self._add_note(state, "Starting hybrid content indexing...")
        state = indexing_node(state)

        # Mark preface as complete
        state["preface_complete"] = True

        # Initialize agent results list if not exists
        if "agent_results" not in state:
            state["agent_results"] = []

        # Store preface results in context_data
        summary_data = state.get("doc_summary", {})
        indexing_data = self._read_context_data(state, "indexing") or {}

        self._write_context_data(state, {
            "completed": True,
            "summary": {
                "pages": summary_data.get("pages", 0),
                "sections": len(summary_data.get("sections", [])),
                "tables": summary_data.get("counts", {}).get("tables", 0),
                "figures": summary_data.get("counts", {}).get("figures", 0)
            },
            "indexing": {
                "pages_processed": indexing_data.get("pages_processed", 0),
                "has_structured_content": indexing_data.get("content_summary", {}).get("has_structured_content", False),
                "output_path": indexing_data.get("output_path", "")
            }
        })

        # Store trace data
        self._write_trace_data(state, {
            "completed": True,
            "steps_executed": ["summarization", "indexing"],
            "summary_pages": summary_data.get("pages", 0),
            "indexing_pages": indexing_data.get("pages_processed", 0)
        }, {
            "execution_order": ["summary", "indexing"],
            "both_completed": True
        })

        self._add_note(state, "Preface complete: document summarized and indexed")

        return state


# Create instance for backward compatibility
preface_node = PrefaceNode()


"""
Legacy index_document function moved to lg_magent_mvp/indexing.py and replaced
with new hybrid parser+LLM approach in IndexingNode.
"""
