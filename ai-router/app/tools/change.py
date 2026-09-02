"""Architect adapter for the team's real change-detection API."""

from app.schemas import ToolRequest, ToolResult


def _has_bi_temporal_context(request: ToolRequest) -> bool:
    """Check for a minimal, explicit indication that the pair represents time T1/T2."""
    if request.parameters.get("image_configuration") == "bi_temporal":
        return True

    roles = {str(image.metadata.get("temporal_role", "")).lower() for image in request.images}
    return roles == {"t1", "t2"}


def run_change_detection_tool(request: ToolRequest) -> ToolResult:
    """Validate adapter inputs and delegate inference to the real ML function."""
    if len(request.images) != 2:
        return ToolResult(
            status="error",
            error="change_detection requires exactly 2 images representing T1 and T2.",
        )

    if not _has_bi_temporal_context(request):
        return ToolResult(
            status="error",
            error=(
                "change_detection requires bi-temporal context. Set "
                "parameters['image_configuration'] to 'bi_temporal' or set "
                "image metadata temporal_role values to 't1' and 't2'."
            ),
        )

    from satquery.inference.api import run_change_detection

    result = run_change_detection(
        request.images[0].metadata["array"], request.images[1].metadata["array"]
    )
    return ToolResult(status="success", data={"inference_result": result})
