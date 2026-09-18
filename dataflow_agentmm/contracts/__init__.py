"""Stable data and interface contracts for the agent runtime.

Runtime orchestration, persistence, provider adapters, and host integrations
belong in the parent package. This namespace is the canonical import boundary
for values exchanged between those implementations.
"""

from .agent import (
    Content,
    ContentLimits,
    ImageContent,
    Message,
    TextContent,
    ToolError,
    ToolResult,
    ToolSpec,
    content_from_dict,
    validate_content,
)
from .environment import (
    Env,
    EnvironmentSpec,
    ReplayVerifier,
    RuleSpec,
    Scenario,
    VerificationCheck,
    VerificationResult,
    close_env,
    start_env,
)
from .task import (
    JudgeCriterion,
    JudgeReference,
    TASK_SCHEMA_VERSION,
    ReplayVerifierFactory,
    ReplayVerifierResolver,
    Task,
    TaskResolver,
    validate_task_id,
)
from .trajectory import EpisodeStep, Trajectory, utc_now

__all__ = [
    "Content",
    "ContentLimits",
    "Env",
    "EnvironmentSpec",
    "EpisodeStep",
    "ImageContent",
    "JudgeCriterion",
    "JudgeReference",
    "Message",
    "ReplayVerifier",
    "ReplayVerifierFactory",
    "ReplayVerifierResolver",
    "RuleSpec",
    "Scenario",
    "TASK_SCHEMA_VERSION",
    "Task",
    "TaskResolver",
    "TextContent",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "Trajectory",
    "VerificationCheck",
    "VerificationResult",
    "close_env",
    "content_from_dict",
    "start_env",
    "utc_now",
    "validate_content",
    "validate_task_id",
]
