"""Always-available episode completion tool."""

from __future__ import annotations

from ..contracts import ToolSpec


FINISH_TOOL_SPEC = ToolSpec(
    name="finish",
    description=(
        "Finish the episode only after the task has been completed and return "
        "the final answer."
    ),
    operation_type="mutation",
    input_schema={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "minLength": 1,
                "description": "The final answer for the completed task.",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
)


__all__ = ["FINISH_TOOL_SPEC"]
