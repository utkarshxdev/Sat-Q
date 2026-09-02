"""Validated execution and audit boundary for registered specialist tools."""

from __future__ import annotations

from app.registry import get_registered_tool
from app.schemas import AuditEvent, FinalResponse, RoutingDecision, ToolRequest
from app.validator import validate_tool_request


def execute_decision(
    request: ToolRequest, decision: RoutingDecision
) -> FinalResponse:
    """Execute one validated decision and return its result with an audit event."""
    tool = get_registered_tool(decision.selected_tool)
    validation = validate_tool_request(request, decision.selected_tool)
    if not validation.is_valid:
        raise ValueError("Cannot execute request: " + "; ".join(validation.errors))

    inputs = {
        "image_ids": [image.id for image in request.images],
        "modalities": [image.modality.lower() for image in request.images],
    }
    try:
        result = tool.implementation(
            ToolRequest(request.query, request.images, decision.parameters)
        )
    except Exception as error:
        audit = AuditEvent(
            task=request.query,
            tool=tool.name,
            tool_identifier=tool.tool_identifier,
            inputs=inputs,
            parameters=decision.parameters,
            status="error",
            error=str(error),
        )
        return FinalResponse(answer=None, status="error", error=str(error), audit_trace=[audit])

    audit = AuditEvent(
        task=request.query,
        tool=tool.name,
        tool_identifier=tool.tool_identifier,
        inputs=inputs,
        parameters=decision.parameters,
        status=result.status,
        result_summary={"status": result.status, "has_data": bool(result.data)},
        error=result.error,
    )
    return FinalResponse(
        answer=result.answer,
        status=result.status,
        confidence=result.confidence,
        evidence=result.evidence,
        error=result.error,
        audit_trace=[audit],
    )