import json
from typing import TYPE_CHECKING, Dict
from pydantic import BaseModel, Field
from ..models import get_router_llm
from ..messages import create_planner_messages
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState


class PlanResponse(BaseModel):
    """Structured response for planning with reasoning."""

    reasoning: str = Field(
        description="Reasoning for the high-level plan"
    )
    steps: Dict[str, str] = Field(
        description="Dictionary mapping step names to reasoning for each step"
    )
    estimated_priority: str = Field(
        description="Overall priority level: 'high', 'medium', or 'low'"
    )

def safe_json_list(text: str) -> list:
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except:
        return []

class PlanNode(BaseNode):
    """Node for creating execution plans based on document analysis."""

    def __init__(self):
        super().__init__(
            "planner",
            description="Creates an execution plan (steps + priority) from the question and document summary."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        llm = get_router_llm()

        # Get available nodes dynamically
        available_nodes = BaseNode.get_all_available_nodes()
        node_descriptions = BaseNode.get_node_descriptions()

        # Filter out nodes that shouldn't be in plans (like planner itself)
        excluded_nodes = {"planner", "summarize", "approval"}  # These are handled by routing logic
        available_nodes = [node for node in available_nodes if node not in excluded_nodes]

        doc_summary = state.get("doc_summary", {})

        # Create structured LLM with response schema
        structured_llm = llm.with_structured_output(PlanResponse)

        messages = create_planner_messages(
            question=state.get('question', 'No specific question provided'),
            doc_summary=doc_summary,
            available_nodes=available_nodes,
            node_descriptions=node_descriptions
        )

        try:
            plan_response = structured_llm.invoke(messages)

            # Debug: log the raw response
            self._add_note(state, f"Raw LLM response type: {type(plan_response)}")
            if isinstance(plan_response, dict):
                self._add_note(state, f"Response keys: {list(plan_response.keys())}")

            # Handle both dict and Pydantic model responses
            if isinstance(plan_response, dict):
                steps_dict = plan_response.get("steps", {})
                reasoning = plan_response.get("reasoning", "")
                priority = plan_response.get("estimated_priority", "medium")

                # Additional validation
                if not steps_dict:
                    self._add_note(state, "Warning: steps field is empty or missing")

            else:
                steps_dict = plan_response.steps
                reasoning = plan_response.reasoning
                priority = plan_response.estimated_priority

            # Convert dictionary to list of step names for processing
            steps = list(steps_dict.keys()) if isinstance(steps_dict, dict) else []

            # Store the step reasoning for later use
            step_reasoning = steps_dict if isinstance(steps_dict, dict) else {}

            # Validate we have steps
            if not steps:
                raise ValueError("No steps were provided in the response")

        except Exception as e:
            # Fallback to default plan if structured output fails
            self._add_note(state, f"Structured planning failed: {e}, using fallback")
            steps = ["keyword_search", "semantic_search", "finalize"]
            step_reasoning = {
                "keyword_search": "Search for specific terms in the document",
                "semantic_search": "Find semantically related content",
                "finalize": "Compile and summarize results"
            }
            reasoning = "Fallback plan due to LLM error"
            priority = "medium"

        # Filter steps based on summary counts to avoid impossible actions
        counts = (doc_summary or {}).get("counts", {})
        original_steps = steps.copy()

        # if counts.get("tables", 0) == 0:
        #     steps = [s for s in steps if s != "extract_tables"]
        # if counts.get("figures", 0) == 0:
        #     steps = [s for s in steps if s != "analyze_figures"]

        # Ensure finalize is always last if present
        if "finalize" in steps:
            steps = [s for s in steps if s != "finalize"] + ["finalize"]

        # Update step_reasoning to only include remaining steps
        filtered_step_reasoning = {step: step_reasoning.get(step, "") for step in steps}

        # Log filtering if steps were removed
        if len(steps) != len(original_steps):
            removed = set(original_steps) - set(steps)
            self._add_note(state, f"Filtered out steps due to document content: {removed}")

        # Store plan with reasoning (legacy format for backward compatibility)
        state["plan"] = steps
        state["cursor"] = 0
        state["plan_reasoning"] = reasoning
        state["plan_priority"] = priority

        # Store in new context_data format
        plan_context = {
            "steps": steps,
            "reasoning": reasoning,
            "priority": priority,
            "step_reasoning": filtered_step_reasoning
        }
        self._write_context_data(state, plan_context)

        # Store detailed trace data
        plan_trace = {
            "original_steps": original_steps,
            "filtered_steps": steps,
            "removed_steps": list(set(original_steps) - set(steps)) if len(steps) != len(original_steps) else [],
            "reasoning": reasoning,
            "priority": priority,
            "step_reasoning": filtered_step_reasoning,
            "doc_summary_counts": counts,
            "available_nodes": available_nodes
        }
        self._write_trace_data(state, plan_trace, {"planning_method": "structured_llm"})

        # Legacy notes for backward compatibility
        self._add_note(state, f"Plan: {steps}")
        self._add_note(state, f"Reasoning: {reasoning}")
        self._add_note(state, f"Priority: {priority}")

        # Log individual step reasoning
        for step, step_reason in filtered_step_reasoning.items():
            if step_reason:
                self._add_note(state, f"Step '{step}': {step_reason}")

        return state


# Create instance for backward compatibility
plan_node = PlanNode()
