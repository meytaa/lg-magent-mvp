from typing import TypedDict, List, Dict, Any, Optional
import os
import langgraph
from langgraph.graph import StateGraph, END

from .nodes.search import keyword_search_node, semantic_search_node
from .nodes.tables import extract_tables_node
from .nodes.vision import analyze_figures_node
from .nodes.finalize import finalize_node
from .nodes.approval import approval_node
from .nodes.summary import summarize_doc_node
from .nodes.indexing import indexing_node
from .nodes import BaseNode
from .orchestrator import orchestrator_node
from .preface import preface_node
from .config import load_settings, ensure_env, apply_tracing_env, load_env_from_dotenv
from .schemas import Finding, Table, Figure, Report, DocumentMeta, DocSummary, OrchestratorResponse
from .models import get_orchestrator_llm

class AgentState(TypedDict, total=False):
    # Core inputs
    question: str
    doc_path: str

    # Orchestrator state
    orchestrator_history: List[Dict[str, Any]]  # Legacy orchestrator history
    orchestrator_chat_history: List[Dict[str, Any]]  # Full conversation history for chat-based orchestrator
    current_plan: Optional[str]  # High-level plan from orchestrator
    next_action: Optional[Dict[str, Any]]  # Next agent + parameters

    # Preface results
    preface_complete: bool
    document_meta: DocumentMeta
    doc_summary: DocSummary

    # Agent execution results
    agent_results: List[Dict[str, Any]]  # Structured results from each agent

    # Legacy evidence (for backward compatibility)
    evidence: List[Dict[str, Any]]
    findings: List[Finding]
    tables: List[Table]
    figures: List[Figure]

    # State management
    context_data: Dict[str, Any]  # Ordered dict: node_name -> data
    trace_data: Dict[str, Any]    # Ordered dict: node_name -> metadata

    # Control flow
    done: bool
    hops: int
    awaiting_approval: bool
    approved: bool
    last_tool: Optional[str]  # Track which tool/agent was last executed

    # Legacy fields (for backward compatibility)
    notes: List[str]  # Legacy notes field
    narrative: Optional[str]  # Legacy narrative field

    # Final outputs
    answer: Optional[str]
    report: Optional[Report]

def with_hop(fn):
    def _inner(s: AgentState) -> AgentState:
        s["hops"] = s.get("hops", 0) + 1
        return fn(s)
    return _inner

def orchestrator_router(state: AgentState) -> str:
    """Router for orchestrator-based architecture"""
    settings = load_settings()

    # Debug logging
    last_tool = state.get("last_tool")
    next_action = state.get("next_action")
    print(f"[ROUTER DEBUG] last_tool: {last_tool}, next_action: {next_action}, hops: {state.get('hops', 0)}")

    # Safety guards
    if state.get("hops", 0) >= settings.max_hops:
        print(f"[ROUTER DEBUG] Max hops reached, routing to finalize")
        return "finalize" if not state.get("done") else END
    if state.get("awaiting_approval") and not state.get("approved"):
        print(f"[ROUTER DEBUG] Awaiting approval, ending")
        return END
    if state.get("done"):
        print(f"[ROUTER DEBUG] Done flag set, ending")
        return END

    # Start with preface if not complete
    if not state.get("preface_complete"):
        print(f"[ROUTER DEBUG] Preface not complete, routing to preface")
        return "preface"

    # Check if orchestrator has decided on next action
    if next_action and next_action.get("agent"):
        agent_name = next_action["agent"]
        print(f"[ROUTER DEBUG] Orchestrator decided on {agent_name}, routing there")
        # TODO: Softcode the agent names
        # Validate agent name
        if agent_name in ["keyword_search", "semantic_search", "extract_tables",
                         "analyze_figures", "approval", "finalize"]:
            return agent_name
        else:
            # Invalid agent, go to finalize
            print(f"[ROUTER DEBUG] Invalid agent {agent_name}, routing to finalize")
            return "finalize"

    # Check if we just came from an agent (not orchestrator or preface)
    # If so, go to orchestrator for next decision
    if last_tool and last_tool not in ["orchestrator", "preface"]:
        print(f"[ROUTER DEBUG] Just came from {last_tool}, routing to orchestrator for next decision")
        return "orchestrator"

    # If we're here and no next action, we might be in a loop
    # Check orchestrator history to prevent infinite loops
    history = state.get("orchestrator_history", [])
    if len(history) > 10:  # Prevent too many orchestrator calls
        print(f"[ROUTER DEBUG] Too many orchestrator calls ({len(history)}), routing to finalize")
        return "finalize"

    # Default to orchestrator for decision making
    print(f"[ROUTER DEBUG] Default routing to orchestrator")
    return "orchestrator"


# Old router functions removed - using orchestrator-based architecture

def _add_nodes_and_edges(g):
    """Helper function to add all nodes and edges to a StateGraph"""
    # Core nodes
    g.add_node("preface", with_hop(preface_node))
    g.add_node("orchestrator", with_hop(orchestrator_node))

    # Preface component nodes (can be called independently)
    g.add_node("summarize", with_hop(summarize_doc_node))
    g.add_node("indexing", with_hop(indexing_node))

    # Agent nodes
    g.add_node("keyword_search", with_hop(keyword_search_node))
    g.add_node("semantic_search", with_hop(semantic_search_node))
    g.add_node("extract_tables", with_hop(extract_tables_node))
    g.add_node("analyze_figures", with_hop(analyze_figures_node))
    g.add_node("approval", with_hop(approval_node))
    g.add_node("finalize", with_hop(finalize_node))

    # Start with preface
    g.set_entry_point("preface")

    # All nodes route through orchestrator
    for node in ["preface", "orchestrator", "summarize", "indexing", "keyword_search", "semantic_search",
                 "extract_tables", "analyze_figures", "approval", "finalize"]:
        g.add_conditional_edges(source=node, path=orchestrator_router)


def build_orchestrator_graph(checkpointer=None):
    """Build graph with orchestrator-based architecture"""
    g = StateGraph(AgentState)
    _add_nodes_and_edges(g)
    return g.compile(checkpointer=checkpointer)


def compile_graph_with_settings(settings=None):
    """Compile orchestrator graph with or without checkpointer based on settings."""
    from .config import load_settings
    settings = settings or load_settings()

    if settings.use_memory:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            print("Using in-memory checkpointer (session-only persistence)")
            return build_orchestrator_graph(checkpointer=checkpointer)

        except ImportError:
            print("No checkpointer available, running without persistence")
            return build_orchestrator_graph()
        except Exception as e:
            print(f"Error setting up checkpointer: {e}")
            return build_orchestrator_graph()
    else:
        return build_orchestrator_graph()

if __name__ == "__main__":
    # Load .env to avoid manual exports
    load_env_from_dotenv()

    # Initialize settings and tracing (Steps 1–2)
    settings = load_settings()
    apply_tracing_env(settings)
    ensure_env(settings)

    # Compile with or without checkpointer (persistence)
    graph = compile_graph_with_settings(settings)
    out = graph.invoke({
        "question": "What are the key findings from the latest report?",
        "doc_path": "data/Amaryllis_chiropractic_report.pdf",
    }, config={"configurable": {"thread_id": "demo"}})
    # Print answer if available
    if "answer" in out and out["answer"]:
        print("ANSWER:")
        print(out["answer"])
    else:
        print("ERROR: No answer was generated")
        print("Final state keys:", list(out.keys()))

    # Print trace information
    print("\nTrace:", *out.get("notes", []), sep="\n- ")

    # Print orchestrator chat history if available
    if "orchestrator_chat_history" in out:
        print(f"\nOrchestrator Chat History ({len(out['orchestrator_chat_history'])} entries):")
        for i, entry in enumerate(out["orchestrator_chat_history"], 1):
            role = entry.get("role", "unknown").upper()
            content_preview = entry.get("content", "")[:100] + "..." if len(entry.get("content", "")) > 100 else entry.get("content", "")
            print(f"  {i}. {role}: {content_preview}")
