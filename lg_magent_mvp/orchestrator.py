from typing import Dict, Any
from .schemas import OrchestratorResponse
from .models import get_orchestrator_llm
from .messages import create_orchestrator_messages
from .nodes import BaseNode


def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Central orchestrator that analyzes state and decides next actions"""

    # Initialize orchestrator chat history if not present
    if "orchestrator_chat_history" not in state:
        state["orchestrator_chat_history"] = []

    # Get available agents
    available_agents = BaseNode.get_all_available_nodes()
    excluded_agents = {"preface", "orchestrator", "finalize", "summarize", "indexing"}
    agent_options = [agent for agent in available_agents if agent not in excluded_agents]

    # Build context for orchestrator with full chat history
    context = {
        "question": state.get("question", ""),
        "preface_results": {
            "doc_summary": state.get("doc_summary", {}),
            "document_meta": state.get("document_meta", {})
        },
        "orchestrator_chat_history": state["orchestrator_chat_history"],
        "available_agents": agent_options,
        "current_findings": len(state.get("findings", [])),
        "current_evidence": len(state.get("evidence", [])),
        "hops": state.get("hops", 0)
    }
    
    # Get LLM decision with structured output
    llm = get_orchestrator_llm()
    messages = create_orchestrator_messages(context)

    # Use structured output to get response directly in the correct schema
    structured_llm = llm.with_structured_output(OrchestratorResponse)

    try:
        orchestrator_response = structured_llm.invoke(messages)

        # Update state (store minimal info to avoid circular references)
        agent_to_call = orchestrator_response.get("agent_to_call")
        agent_name = "none"
        if agent_to_call:
            agent_name = agent_to_call["name"]

        # Check if orchestrator provided final answer (process complete)
        final_answer = orchestrator_response.get("final_answer")
        process_complete = orchestrator_response.get("process_complete", False)

        # Build the FULL orchestrator response content for chat history
        thoughts = orchestrator_response.get("thoughts", "")

        if process_complete and final_answer:
            # Orchestrator has provided final answer and signaled completion
            orchestrator_content = f"THOUGHTS: {thoughts}\n\nFINAL ANSWER: {final_answer}\n\nPROCESS COMPLETE: True"
            state["done"] = True
            state["answer"] = final_answer
        else:
            # Regular agent selection - include full thoughts and decision
            orchestrator_content = f"THOUGHTS: {thoughts}\n\n"

            if agent_to_call:
                orchestrator_content += f"AGENT SELECTED: {agent_to_call['name']}\n"
                orchestrator_content += f"PARAMETERS: {agent_to_call['input_arguments']}\n"
                orchestrator_content += f"REASONING: Based on my analysis, I'm selecting {agent_to_call['name']} to gather more information."
            else:
                orchestrator_content += "DECISION: Analysis complete - no further agents needed\n"
                orchestrator_content += "REASONING: I have sufficient information to proceed with finalization."

        state["orchestrator_chat_history"].append({
            "role": "assistant",
            "content": orchestrator_content,
            "full_response": orchestrator_response,  # Store the complete LLM response
            "thoughts": thoughts,
            "agent_to_call": agent_to_call,
            "final_answer": final_answer,
            "process_complete": process_complete
        })

        # Convert new format to old format for compatibility
        if agent_to_call and not process_complete:
            state["next_action"] = {
                "agent": agent_to_call["name"],
                "parameters": agent_to_call["input_arguments"],
                "reasoning": f"Orchestrator decided to call {agent_name}"
            }
        else:
            state["next_action"] = None

        # Set done flag - only when process is complete or we have a final answer
        if not process_complete:
            if agent_to_call is None:
                # If no agent to call but no final answer provided, this might be an error
                if not final_answer:
                    # Orchestrator decided to end but didn't provide final answer
                    # This could be an LLM error or incomplete response
                    state["done"] = False  # Don't end yet
                    state["next_action"] = None
                    # Add a note about this issue
                    state.setdefault("notes", []).append("Orchestrator ended without final answer - may need manual intervention")
                else:
                    # We have a final answer, safe to end
                    state["done"] = True
            else:
                # We have an agent to call, continue
                state["done"] = False

        # Set last_tool to orchestrator so router knows we just came from orchestrator
        state["last_tool"] = "orchestrator"

        # Add to trace
        if "trace_data" not in state:
            state["trace_data"] = {}

        state["trace_data"]["orchestrator"] = {
            "thoughts": orchestrator_response.get("thoughts"),
            "agent_to_call": orchestrator_response.get("agent_to_call")
        }

        # Add note
        if "notes" not in state:
            state["notes"] = []

        action_desc = "Planning complete" if state["done"] else f"Next: {agent_name}"
        state["notes"].append(f"Orchestrator: {action_desc}")

    except Exception as e:
        # Fallback behavior
        state["notes"] = state.get("notes", [])
        state["notes"].append(f"Orchestrator error: {e}")

        # Provide a fallback answer if we don't have one
        if not state.get("answer"):
            # Generate a basic summary from available evidence
            chat_history = state.get("orchestrator_chat_history", [])
            agent_results = [entry for entry in chat_history if entry.get("role") == "user" and "AGENT RESULT" in entry.get("content", "")]

            if agent_results:
                fallback_answer = f"Analysis completed with {len(agent_results)} agent results. "
                fallback_answer += "Due to an orchestrator error, a detailed final answer could not be generated. "
                fallback_answer += "Please review the individual agent results for findings."
            else:
                fallback_answer = "Analysis could not be completed due to orchestrator error. No agent results were obtained."

            state["answer"] = fallback_answer

        state["done"] = True  # End on error
        # Set last_tool even on error
        state["last_tool"] = "orchestrator"

    return state



