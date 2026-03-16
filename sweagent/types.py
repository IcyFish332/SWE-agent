"""This file has types/dataclass definitions that are used in the SWE agent
for exchanging data between different modules/functions/classes.
They oftentimes cannot be defined in the same file where they are used
because of circular dependencies.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class StepOutput(BaseModel):
    query: list[dict[str, Any]] | list[int] = Field(default_factory=list)
    thought: str = ""
    action: str = ""
    output: str = ""
    observation: str = ""
    execution_time: float = 0.0
    done: bool = False
    exit_status: int | str | None = None
    submission: str | None = None
    state: dict[str, str] = {}
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_ids: list[str] | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None
    output_tokens: list[int] = []
    rollout_log_probs: list[float] = []
    rollout_routed_experts: list[list[int]] = []

    """State of the environment at the end of the step"""
    extra_info: dict[str, Any] = {}

    def to_template_format_dict(self) -> dict[str, str | int | float | bool | None]:
        """Used for formatting (error) prompt templates"""
        out = {}
        for k, v in self.model_dump().items():
            if k in ("tool_calls", "tool_call_ids", "state"):
                continue
            out[k] = v
        out |= self.state
        return out


class TrajectoryStep(TypedDict):
    action: str
    observation: str
    response: str | None
    state: dict[str, str]
    thought: str | None
    execution_time: float
    query: list[dict[str, Any]] | list[int]
    extra_info: dict[str, Any]
    reasoning_content: str | None
    output_tokens: list[int] | None
    rollout_log_probs: list[float] | None
    rollout_routed_experts: list[list[int]] | None


# required fields go here
class _HistoryItem(TypedDict):
    role: str
    content: str | list[dict[str, Any]]
    message_type: Literal["thought", "action", "observation"]


# see _HistoryItem for required fields
class HistoryItem(_HistoryItem, total=False):
    agent: str
    is_demo: bool
    thought: str
    action: str | None
    tool_calls: list[dict[str, str]] | None
    tool_call_ids: list[str] | None
    tags: list[str]
    cache_control: dict[str, Any] | None
    thinking_blocks: list[dict[str, Any]] | None
    output_tokens: list[int] | None

    """HistoryProcessors can add these tags to enable special processing"""


History = list[HistoryItem]
Trajectory = list[TrajectoryStep]


# todo: Make this actually have the dataclasses instead of dict versions
class AgentInfo(TypedDict, total=False):
    # same as `APIStats` from models.py
    model_stats: dict[str, float]
    exit_status: str | None
    submission: str | None
    # same as `ReviewerResult`
    review: dict[str, Any]
    edited_files30: str
    edited_files50: str
    edited_files70: str
    # only if summarizer is used
    summarizer: dict
    swe_agent_hash: str
    swe_agent_version: str
    swe_rex_version: str
    swe_rex_hash: str
    error_logs: list[dict[str, Any]]
    rollout_log_probs: list[float] | None
    rollout_routed_experts: list[list[int]] | None
    response_turn: int | None


class AgentRunResult(BaseModel):
    info: AgentInfo
    trajectory: Trajectory
