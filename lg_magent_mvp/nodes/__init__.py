from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
import inspect

if TYPE_CHECKING:
    from ..app import AgentState

from ..state_manager import StateManager, write_context, write_trace, read_context, read_trace


class BaseNode(ABC):
    """Base class for all LangGraph nodes in the system.

    Provides common functionality like state management, logging, and error handling.
    All nodes should inherit from this class and implement the execute method.

    Description handling: prefer explicit `self.description` over class docstrings.
    Subclasses should set a human-readable description either by passing it to the
    constructor or by assigning `self.description` in their `__init__`.
    """

    def __init__(self, node_name: str, description: Optional[str] = None):
        """Initialize the base node.

        Args:
            node_name: The name of the node for logging and state tracking
            description: Optional human-readable description of the node
        """
        self.node_name = node_name
        self.description: Optional[str] = description

    def __call__(self, state: "AgentState") -> "AgentState":
        """Main entry point for the node. Handles common setup and teardown."""
        try:
            # Initialize state management structures
            StateManager.initialize_state_data(state)

            # Migrate legacy state if needed (for backward compatibility)
            self._migrate_legacy_state(state)

            # Add node execution to notes (legacy support)
            self._add_note(state, f"Starting {self.node_name}")

            # Also add to trace data
            self._write_trace_data(state, {"status": "starting"}, {"execution_phase": "start"})

            # Execute the node-specific logic
            result_state = self.execute(state)

            # Set last_tool if not already set by the node
            if "last_tool" not in result_state or not result_state["last_tool"]:
                result_state["last_tool"] = self.node_name

            # Add completion to trace data (merge with existing data if present)
            existing_trace = self._read_trace_data(result_state)
            if existing_trace and isinstance(existing_trace.get("data"), dict):
                # Merge completion status with existing data
                merged_data = existing_trace["data"].copy()
                merged_data["status"] = "completed"
                merged_metadata = existing_trace.get("metadata", {}).copy()
                merged_metadata["execution_phase"] = "end"
                self._write_trace_data(result_state, merged_data, merged_metadata)
            else:
                # No existing data, write completion status
                self._write_trace_data(result_state, {"status": "completed"}, {"execution_phase": "end"})

            return result_state

        except Exception as e:
            # Handle errors gracefully
            self._add_note(state, f"Error in {self.node_name}: {str(e)}")
            self._write_trace_data(state, {"status": "error", "error": str(e)}, {"execution_phase": "error"})
            return state

    @abstractmethod
    def execute(self, state: "AgentState") -> "AgentState":
        """Execute the node-specific logic.

        Args:
            state: The current agent state

        Returns:
            The updated agent state
        """
        pass

    def _add_note(self, state: "AgentState", note: str) -> None:
        """Add a note to the state."""
        state.setdefault("notes", []).append(note)

    def _add_evidence(self, state: "AgentState", evidence_type: str, data: Any) -> None:
        """Add evidence to the state."""
        state.setdefault("evidence", []).append({"type": evidence_type, "data": data})

    def _add_agent_result(self, state: "AgentState", success: bool, data: Dict[str, Any], error: Optional[str] = None) -> None:
        """Add structured agent result for orchestrator."""
        result = {
            "agent": self.node_name,
            "success": success,
            "data": data,
            "error": error,
            "metadata": {
                "timestamp": "now",  # Can use actual timestamp
                "hops": state.get("hops", 0)
            }
        }

        state.setdefault("agent_results", []).append(result)

        # Also add to orchestrator chat history if it exists
        if "orchestrator_chat_history" in state:
            # Format agent result for chat history based on agent type
            result_text = f"AGENT RESULT - {self.node_name}: {'Success' if success else 'Failed'}\n"

            if not success:
                # Handle failed agent results
                result_text += f"Error: {error or 'Unknown error occurred'}\n"
                if data and "error" in data:
                    result_text += f"Details: {data['error']}\n"
            elif success and data:
                # Handle keyword search results specifically
                if self.node_name == "keyword_search" and "results" in data:
                    result_text += f"Found {data.get('hits', 0)} total hits for keywords: {data.get('keywords_searched', [])}\n\n"
                    if data.get('hits', 0) > 0:
                        result_text += "Key findings with specific references:\n"
                        for i, hit in enumerate(data["results"][:5], 1):  # Show top 5 results
                            page = hit.get('page', 'Unknown')
                            snippet = hit.get('snippet', hit.get('text', ''))
                            doc = hit.get('doc', 'document')
                            span = hit.get('span', [])
                            # Truncate snippet to reasonable length but keep it informative
                            if len(snippet) > 200:
                                snippet = snippet[:200] + "..."
                            # Include more detailed reference information
                            ref_info = f"Page {page}"
                            if span and len(span) >= 2:
                                ref_info += f" (text span {span[0]}-{span[1]})"
                            result_text += f"  {i}. {ref_info}: {snippet}\n"
                    else:
                        result_text += "No matching keywords found in the document.\n"

                # Handle semantic search results specifically
                elif self.node_name == "semantic_search" and "results" in data:
                    result_text += f"Found {data.get('hits', 0)} semantically similar passages for query: '{data.get('query', '')}'\n\n"
                    if data.get('hits', 0) > 0:
                        result_text += "Key findings with specific references:\n"
                        for i, hit in enumerate(data["results"][:5], 1):  # Show top 5 results
                            page = hit.get('page', 'Unknown')
                            snippet = hit.get('snippet', '')
                            score = hit.get('score', 0)
                            chunk_id = hit.get('chunk_id', '')
                            span = hit.get('span', [])
                            # Truncate snippet to reasonable length but keep it informative
                            if len(snippet) > 200:
                                snippet = snippet[:200] + "..."
                            # Include more detailed reference information
                            ref_info = f"Page {page} (similarity: {score:.3f})"
                            if chunk_id:
                                ref_info += f" [chunk: {chunk_id}]"
                            if span and len(span) >= 2:
                                ref_info += f" (text span {span[0]}-{span[1]})"
                            result_text += f"  {i}. {ref_info}: {snippet}\n"
                    else:
                        result_text += "No semantically similar content found for the query.\n"

                # Handle extract_tables agent results specifically
                elif self.node_name == "extract_tables":
                    table_count = data.get('count', 0)
                    extraction_successful = data.get('extraction_successful', False)

                    if extraction_successful and table_count > 0:
                        result_text += f"Successfully extracted {table_count} tables with specific references\n\n"
                        tables = data.get('tables', [])
                        result_text += "Tables found with references:\n"
                        for i, table in enumerate(tables[:3], 1):  # Show first 3 tables
                            page = table.get('page', 'Unknown')
                            table_id = table.get('table_id', f'T{page}-{i}')
                            rows = len(table.get('rows', []))
                            title = table.get('title', 'No title')
                            headers = table.get('headers', [])
                            result_text += f"  {i}. Table {table_id} on page {page}: {title}\n"
                            result_text += f"     {rows} rows, {len(headers)} columns\n"
                            if headers:
                                headers_preview = ', '.join(headers[:3])
                                if len(headers) > 3:
                                    headers_preview += f" (and {len(headers)-3} more)"
                                result_text += f"     Headers: {headers_preview}\n"
                        if len(tables) > 3:
                            result_text += f"  ... and {len(tables) - 3} more tables\n"
                    else:
                        result_text += "No tables found in the document\n"

                # Handle analyze_figures agent results specifically
                elif self.node_name == "analyze_figures":
                    success = data.get('success', False)
                    figures_analyzed = data.get('figures_analyzed', 0)
                    figure_ids = data.get('figure_ids', [])
                    analyses = data.get('analyses', [])
                    overall_insights = data.get('overall_insights', '')

                    if success and figures_analyzed > 0:
                        result_text += f"Successfully analyzed {figures_analyzed} figures with specific references: {', '.join(figure_ids)}\n\n"

                        # Show individual figure analyses with detailed references
                        result_text += "Figure Analysis Results with References:\n"
                        for i, analysis in enumerate(analyses[:3], 1):  # Show first 3 analyses
                            figure_id = analysis.get('figure_id', 'Unknown')
                            confidence = analysis.get('confidence', 'unknown')
                            key_findings = analysis.get('key_findings', [])
                            relevance = analysis.get('relevance_to_question', '')

                            # Extract page number from figure_id if possible (e.g., fig_page1_1 -> page 1)
                            page_num = 'Unknown'
                            if 'page' in figure_id:
                                try:
                                    page_num = figure_id.split('page')[1].split('_')[0]
                                except:
                                    page_num = 'Unknown'
                            analysis_text = analysis.get('analysis', '')[:100] + "..." if len(analysis.get('analysis', '')) > 100 else analysis.get('analysis', '')

                            result_text += f"  {i}. Figure {figure_id} on page {page_num} (Confidence: {confidence})\n"
                            result_text += f"     Analysis: {analysis_text}\n"
                            if key_findings:
                                result_text += f"     Key findings: {', '.join(key_findings[:2])}\n"
                            if relevance:
                                relevance_preview = relevance[:80] + "..." if len(relevance) > 80 else relevance
                                result_text += f"     Relevance: {relevance_preview}\n"
                            result_text += "\n"

                        if len(analyses) > 3:
                            result_text += f"  ... and {len(analyses) - 3} more analyses\n\n"

                        # Show overall insights
                        if overall_insights:
                            insights_preview = overall_insights[:150] + "..." if len(overall_insights) > 150 else overall_insights
                            result_text += f"Overall Insights: {insights_preview}\n"
                    else:
                        error_msg = data.get('error', 'Unknown error')
                        requested_ids = data.get('figure_ids', [])
                        if requested_ids:
                            result_text += f"Failed to analyze figures {', '.join(requested_ids)}: {error_msg}\n"
                        else:
                            result_text += f"Figure analysis failed: {error_msg}\n"

                # Handle approval agent results specifically
                elif self.node_name == "approval":
                    approval_mode = data.get('approval_mode', 'unknown')
                    approved = data.get('approved', False)
                    awaiting = data.get('awaiting', False)

                    result_text += f"Approval mode: {approval_mode}\n"
                    result_text += f"Status: {'Approved' if approved else 'Awaiting approval' if awaiting else 'Not approved'}\n"

                    if data.get('auto_approved'):
                        result_text += "Auto-approval granted - proceeding with workflow\n"
                    elif data.get('user_approved'):
                        result_text += "User approval received - proceeding with workflow\n"
                    elif awaiting:
                        result_text += "Workflow paused - waiting for user approval\n"

                # Handle finalize agent results specifically
                elif self.node_name == "finalize" and "narrative" in data:
                    result_text += f"Final analysis completed with confidence: {data.get('confidence', 'N/A')}/10\n"
                    result_text += f"Conclusion obtained: {data.get('conclusion_obtained', 'Unknown')}\n"
                    result_text += f"Evidence processed: {data.get('evidence_count', 0)} pieces\n\n"

                    narrative = data.get('narrative', '')
                    # Show first part of narrative
                    if len(narrative) > 300:
                        result_text += f"Final Analysis Summary:\n{narrative[:300]}...\n\n"
                    else:
                        result_text += f"Final Analysis:\n{narrative}\n\n"

                    result_text += "Ready for orchestrator to provide final answer and complete process."

                # Handle other agent types with generic formatting
                elif "findings" in data and data["findings"]:
                    result_text += f"Findings: {len(data['findings'])} new findings\n"
                    # Include first few findings as examples
                    for i, finding in enumerate(data["findings"][:3], 1):
                        finding_text = finding.get('text', '') if isinstance(finding, dict) else str(finding)
                        result_text += f"  {i}. {finding_text[:100]}...\n"

                # Add other data types
                if "hits" in data and self.node_name != "keyword_search":  # Avoid duplicate for keyword search
                    result_text += f"Search hits: {data['hits']}\n"
                if "evidence" in data and data["evidence"]:
                    result_text += f"Evidence: {len(data['evidence'])} pieces of evidence\n"
                if "tables" in data and data["tables"]:
                    result_text += f"Tables: {len(data['tables'])} tables extracted\n"
                if "figures" in data and data["figures"]:
                    result_text += f"Figures: {len(data['figures'])} figures analyzed\n"
            elif error:
                result_text += f"Error: {error}\n"

            result_text += "\nWhat should be the next action?"

            # Add result_text to the result object for later access
            result["result_text"] = result_text

            # Add to chat history as user message (agent result)
            state["orchestrator_chat_history"].append({
                "role": "user",
                "content": result_text,
                "agent_result": result
            })

    def _get_doc_path(self, state: "AgentState") -> Optional[str]:
        """Get the document path from state, with validation."""
        doc_path = state.get("doc_path")
        if not doc_path:
            self._add_note(state, f"No doc_path provided; skipping {self.node_name}")
            return None
        return doc_path

    def _validate_required_fields(self, state: "AgentState", fields: List[str]) -> bool:
        """Validate that required fields are present in the state.

        Args:
            state: The agent state to validate
            fields: List of required field names

        Returns:
            True if all fields are present, False otherwise
        """
        missing_fields = [field for field in fields if field not in state]
        if missing_fields:
            self._add_note(state, f"Missing required fields for {self.node_name}: {missing_fields}")
            return False
        return True

    # New state management methods
    def _write_context_data(self, state: "AgentState", data: Any) -> None:
        """Write data to context_data for this node."""
        write_context(state, self.node_name, data)

    def _write_trace_data(self, state: "AgentState", data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write data to trace_data for this node with optional metadata."""
        write_trace(state, self.node_name, data, metadata)

    def _read_context_data(self, state: "AgentState", node_name: Optional[str] = None) -> Any:
        """Read context data for this node or another specified node."""
        target_node = node_name or self.node_name
        return read_context(state, target_node)

    def _read_trace_data(self, state: "AgentState", node_name: Optional[str] = None) -> Any:
        """Read trace data for this node or another specified node."""
        target_node = node_name or self.node_name
        return read_trace(state, target_node)

    def _migrate_legacy_state(self, state: "AgentState") -> None:
        """Migrate legacy state data to new format if needed."""
        StateManager.migrate_legacy_state(state)

    @classmethod
    def get_all_available_nodes(cls) -> List[str]:
        """Get a list of all available node names from registered BaseNode subclasses.

        Returns:
            List of node names that can be used in plans
        """
        available_nodes = []

        # Get all subclasses of BaseNode
        for subclass in cls.__subclasses__():
            # Create a temporary instance to get the node name
            try:
                # Try to create instance with dummy node_name
                temp_instance = subclass("temp", "temp description")
                available_nodes.append(temp_instance.node_name)
            except Exception:
                # If constructor fails, try to infer from class name
                class_name = subclass.__name__
                if class_name.endswith('Node'):
                    # Convert CamelCase to snake_case
                    node_name = ''.join(['_' + c.lower() if c.isupper() and i > 0 else c.lower()
                                       for i, c in enumerate(class_name[:-4])])  # Remove 'Node' suffix
                    available_nodes.append(node_name)

        return sorted(available_nodes)

    @classmethod
    def get_node_descriptions(cls) -> Dict[str, str]:
        """Get descriptions of all available nodes.

        Returns:
            Dictionary mapping node names to their descriptions. Prefers an
            explicit `description` attribute on the instance; falls back to
            the first line of the class docstring if absent.
        """
        descriptions = {}

        for subclass in cls.__subclasses__():
            try:
                temp_instance = subclass("temp", "temp description")
                node_name = temp_instance.node_name
                # Prefer explicit instance description; fallback to class docstring
                if getattr(temp_instance, "description", None):
                    description = str(temp_instance.description).strip()
                else:
                    doc = subclass.__doc__ or ""
                    description = doc.split('\n')[0].strip() if doc else f"Node: {node_name}"
                descriptions[node_name] = description
            except Exception:
                continue

        return descriptions


# Import all node classes to ensure they're registered as subclasses
def _import_all_nodes():
    """Import all node modules to ensure BaseNode subclasses are registered."""
    try:
        from . import approval, finalize, indexing, plan, search, summary, tables, vision
    except ImportError:
        # Some imports might fail in certain environments
        pass

# Import nodes when module is loaded
_import_all_nodes()

# Export the base class for external use
__all__ = ["BaseNode"]
