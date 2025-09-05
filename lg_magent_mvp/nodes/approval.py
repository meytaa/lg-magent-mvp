from __future__ import annotations

from typing import TYPE_CHECKING
from ..config import load_settings
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState


class ApprovalNode(BaseNode):
    """Node for handling approval workflows."""

    def __init__(self):
        super().__init__(
            "approval",
            description="Gate node that pauses for user approval before finalizing when required."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        # Clear next_action since we're consuming it (if it was meant for us)
        next_action = state.get("next_action", {})
        if next_action and next_action.get("agent") == "approval":
            state["next_action"] = None

        settings = load_settings()
        mode = settings.approvals

        if mode != "pause-before-finalize":
            # Store in context_data
            approval_context = {"mode": "auto", "approved": True, "awaiting": False}
            self._write_context_data(state, approval_context)

            # Store in trace_data
            self._write_trace_data(state, approval_context, {
                "approval_mode": mode,
                "auto_approved": True
            })

            # Add structured result for orchestrator
            result_data = {
                "approval_mode": mode,
                "approved": True,
                "awaiting": False,
                "auto_approved": True
            }
            self._add_agent_result(state, True, result_data)

            self._add_note(state, "approvals=auto")
            return state

        if state.get("approved"):
            # Store approval in context_data
            approval_context = {"mode": "manual", "approved": True, "awaiting": False}
            self._write_context_data(state, approval_context)

            # Store in trace_data
            self._write_trace_data(state, approval_context, {
                "approval_mode": mode,
                "user_approved": True
            })

            # Add structured result for orchestrator
            result_data = {
                "approval_mode": mode,
                "approved": True,
                "awaiting": False,
                "user_approved": True
            }
            self._add_agent_result(state, True, result_data)

            self._add_note(state, "approved=true")
            return state

        # request approval and mark awaiting
        state["awaiting_approval"] = True

        # Store awaiting state in context_data
        approval_context = {"mode": "manual", "approved": False, "awaiting": True}
        self._write_context_data(state, approval_context)

        # Store in trace_data
        self._write_trace_data(state, approval_context, {
            "approval_mode": mode,
            "awaiting_user_input": True
        })

        # Add structured result for orchestrator
        result_data = {
            "approval_mode": mode,
            "approved": False,
            "awaiting": True,
            "awaiting_user_input": True
        }
        self._add_agent_result(state, True, result_data)

        self._add_note(state, "awaiting_approval=true")
        return state


# Create instance for backward compatibility
approval_node = ApprovalNode()
