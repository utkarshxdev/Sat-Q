"""Shared data contracts for the SatQuery orchestration layer.

These models intentionally contain no routing, model, or API logic.  They
define the stable inputs and outputs that future components will exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


ToolName = Literal[
    "single_image_vqa",
    "run_change_detection",
    "run_optical_sar_fusion",
]
ExecutionStatus = Literal["success", "error"]


@dataclass
class ImageContext:
    """Describes one input image without loading or processing its pixels."""

    id: str
    modality: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ImageContext.id must not be empty.")
        if not self.modality.strip():
            raise ValueError("ImageContext.modality must not be empty.")


@dataclass
class ToolRequest:
    """Validated information passed from the orchestrator to a specialist tool."""

    query: str
    images: list[ImageContext]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("ToolRequest.query must not be empty.")


@dataclass
class RoutingDecision:
    """The strict, provider-independent output contract of the routing step."""

    selected_tool: ToolName
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selected_tool, str) or not self.selected_tool.strip():
            raise ValueError("RoutingDecision.selected_tool must be a non-empty string.")
        if not isinstance(self.parameters, dict):
            raise ValueError("RoutingDecision.parameters must be a dictionary.")

        # Import lazily to avoid a module-import cycle: registry imports schemas.
        from app.registry import get_registered_tool

        get_registered_tool(self.selected_tool)


@dataclass
class ToolResult:
    """A normalized result returned by every specialist tool."""

    status: ExecutionStatus
    answer: str | None = None
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ToolResult.confidence must be between 0.0 and 1.0.")
        if self.status == "success" and self.error is not None:
            raise ValueError("A successful ToolResult cannot contain an error.")
        if self.status == "error" and not self.error:
            raise ValueError("An error ToolResult must include an error message.")


@dataclass
class AuditEvent:
    """An observable record of one tool execution, excluding internal reasoning."""

    task: str
    tool: ToolName
    tool_identifier: str
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    status: ExecutionStatus
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("AuditEvent.task must not be empty.")
        if not self.tool_identifier.strip():
            raise ValueError("AuditEvent.tool_identifier must not be empty.")
        if self.status == "success" and self.error is not None:
            raise ValueError("A successful AuditEvent cannot contain an error.")
        if self.status == "error" and not self.error:
            raise ValueError("An error AuditEvent must include an error message.")


@dataclass
class FinalResponse:
    """The response contract returned to the eventual API or user interface."""

    answer: str | None
    status: ExecutionStatus
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    audit_trace: list[AuditEvent] = field(default_factory=list)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("FinalResponse.confidence must be between 0.0 and 1.0.")
        if self.status == "success" and self.error is not None:
            raise ValueError("A successful FinalResponse cannot contain an error.")
        if self.status == "error" and not self.error:
            raise ValueError("An error FinalResponse must include an error message.")
