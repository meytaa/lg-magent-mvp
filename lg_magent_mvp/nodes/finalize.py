from typing import TYPE_CHECKING
from ..models import get_finalize_llm
from ..schemas import default_report
from ..config import load_settings
from ..messages import create_finalize_messages
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState

class FinalizeNode(BaseNode):
    """Node for finalizing the analysis and creating the final report."""

    def __init__(self):
        super().__init__(
            "finalize",
            description="Synthesizes evidence into a concise narrative and structured report."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        # Get orchestrator parameters
        next_action = state.get("next_action", {})
        parameters = next_action.get("parameters", {}) if next_action else {}
        conclusion_obtained = parameters.get("conclusion_obtained", True)
        confidence = parameters.get("confidence", 8)

        # Clear next_action since we're consuming it
        if next_action and next_action.get("agent") == "finalize":
            state["next_action"] = None

        llm = get_finalize_llm()

        # Gather all evidence from orchestrator chat history and legacy evidence
        all_evidence = []

        # Extract evidence from orchestrator chat history (agent results)
        chat_history = state.get("orchestrator_chat_history", [])
        for entry in chat_history:
            if entry.get("role") == "user" and "AGENT RESULT" in entry.get("content", ""):
                all_evidence.append({
                    "type": "agent_result",
                    "content": entry.get("content", ""),
                    "agent_result": entry.get("agent_result", {})
                })

        # Add legacy evidence for backward compatibility
        legacy_evidence = state.get('evidence', [])
        all_evidence.extend(legacy_evidence)

        # Compose a comprehensive final analysis
        messages = create_finalize_messages(
            question=state.get('question', ''),
            evidence=all_evidence
        )
        response = llm.invoke(messages)
        narrative = str(response.content if hasattr(response, 'content') else response)

        # Build a structured report
        pages = 0
        try:
            pages = int(state.get("doc_summary", {}).get("pages", 0))
        except Exception:
            pages = 0

        settings = load_settings()
        report = default_report(
            file=state.get("doc_path", ""),
            pages=pages,
            model_versions={
                "router_model": settings.router_model,
                "finalize_model": settings.finalize_model,
                "vision_model": settings.vision_model,
                "embed_model": settings.embed_model,
            },
            document_meta=state.get("document_meta", {}),
            findings=state.get("findings", []),
            tables=state.get("tables", []),
            figures=state.get("figures", []),
            narrative=narrative,
        )

        # Store in legacy format for backward compatibility
        state["report"] = report
        state["answer"] = narrative
        # Don't set done=True yet - let orchestrator decide when to complete

        # Store in new context_data format
        finalize_context = {
            "report": report,
            "narrative": narrative,
            "completed": False,  # Not fully completed until orchestrator signals
            "conclusion_obtained": conclusion_obtained,
            "confidence": confidence
        }
        self._write_context_data(state, finalize_context)

        # Store detailed trace data
        finalize_trace = {
            "report": report,
            "narrative": narrative,
            "pages": pages,
            "model_versions": {
                "router_model": settings.router_model,
                "finalize_model": settings.finalize_model,
                "vision_model": settings.vision_model,
                "embed_model": settings.embed_model,
            },
            "input_data": {
                "document_meta": state.get("document_meta", {}),
                "findings": state.get("findings", []),
                "tables": state.get("tables", []),
                "figures": state.get("figures", []),
            },
            "completed": False,
            "orchestrator_parameters": {
                "conclusion_obtained": conclusion_obtained,
                "confidence": confidence
            }
        }
        self._write_trace_data(state, finalize_trace, {
            "finalization_method": "default_report",
            "doc_path": state.get("doc_path", "")
        })

        # Add structured result for orchestrator with the final analysis
        result_data = {
            "narrative": narrative,
            "report": report,
            "conclusion_obtained": conclusion_obtained,
            "confidence": confidence,
            "evidence_count": len(all_evidence),
            "ready_for_completion": True
        }
        self._add_agent_result(state, True, result_data)

        self._add_note(state, "finalize analysis completed - awaiting orchestrator completion")
        return state


# Create instance for backward compatibility
finalize_node = FinalizeNode()
