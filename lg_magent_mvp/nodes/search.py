from typing import TYPE_CHECKING
from ..tools.retrievers import keyword_retriever, semantic_retriever
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState

def your_keyword_search(question: str, doc_path: str | None = None) -> list:
    return keyword_retriever(question, doc_path=doc_path)

def your_vector_search(question: str, doc_path: str | None = None) -> list:
    return semantic_retriever(question, doc_path=doc_path)

class KeywordSearchNode(BaseNode):
    """Node for performing keyword-based search."""

    def __init__(self):
        super().__init__(
            "keyword_search",
            description="Finds relevant passages using keyword-based retrieval."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        if not self._validate_required_fields(state, ["question"]):
            self._add_agent_result(state, False, {}, "Missing required fields")
            return state

        doc_path = state.get("doc_path")

        # Get search parameters from orchestrator if available
        next_action = state.get("next_action", {})
        parameters = next_action.get("parameters", {})
        keywords = parameters.get("keywords", [state["question"]])

        # Clear next_action since we're consuming it (if it was meant for us)
        if next_action and next_action.get("agent") == "keyword_search":
            state["next_action"] = None

        # Perform search for each keyword
        all_hits = []
        for keyword in keywords:
            hits = your_keyword_search(keyword, doc_path=doc_path)
            all_hits.extend(hits)

        # Remove duplicates while preserving order
        seen = set()
        unique_hits = []
        for hit in all_hits:
            hit_id = f"{hit.get('page', 0)}_{hit.get('text', '')[:50]}"
            if hit_id not in seen:
                seen.add(hit_id)
                unique_hits.append(hit)

        # Store in legacy format for backward compatibility
        self._add_evidence(state, "keyword_hits", unique_hits)

        # Store in new context_data format
        search_context = {
            "hits": unique_hits,
            "count": len(unique_hits),
            "query": state["question"],
            "keywords": keywords,
            "search_type": "keyword"
        }
        self._write_context_data(state, search_context)

        # Store detailed trace data
        self._write_trace_data(state, search_context, {
            "doc_path": doc_path,
            "search_method": "your_keyword_search",
            "parameters": parameters
        })

        # Add structured result for orchestrator
        result_data = {
            "hits": len(unique_hits),
            "keywords_searched": keywords,
            "results": unique_hits[:5]  # Top 5 results for orchestrator
        }
        self._add_agent_result(state, True, result_data)

        self._add_note(state, f"keyword hits: {len(unique_hits)}")
        return state


class SemanticSearchNode(BaseNode):
    """Node for performing semantic/vector-based search."""

    def __init__(self):
        super().__init__(
            "semantic_search",
            description="Finds relevant passages using semantic/vector retrieval."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        if not self._validate_required_fields(state, ["question"]):
            return state

        # Clear next_action since we're consuming it (if it was meant for us)
        next_action = state.get("next_action", {})
        if next_action and next_action.get("agent") == "semantic_search":
            state["next_action"] = None

        doc_path = state.get("doc_path")
        hits = your_vector_search(state["question"], doc_path=doc_path)

        # Store in legacy format for backward compatibility
        self._add_evidence(state, "semantic_hits", hits)

        # Store in new context_data format
        search_context = {
            "hits": hits,
            "count": len(hits),
            "query": state["question"],
            "search_type": "semantic"
        }
        self._write_context_data(state, search_context)

        # Store detailed trace data
        self._write_trace_data(state, search_context, {
            "doc_path": doc_path,
            "search_method": "your_vector_search"
        })

        # Add structured result for orchestrator
        result_data = {
            "hits": len(hits),
            "query": state["question"],
            "results": hits[:5]  # Top 5 results for orchestrator
        }
        self._add_agent_result(state, True, result_data)

        self._add_note(state, f"semantic hits: {len(hits)}")
        return state


# Create instances for backward compatibility
keyword_search_node = KeywordSearchNode()
semantic_search_node = SemanticSearchNode()
