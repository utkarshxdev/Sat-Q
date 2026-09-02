"""Architect adapter for the team's real optical-SAR fusion API."""

from app.schemas import ToolRequest, ToolResult


_OPTICAL_MODALITIES = {"optical", "multispectral"}


def run_optical_sar_fusion_tool(request: ToolRequest) -> ToolResult:
    """Validate adapter inputs and delegate inference to the real ML function."""
    if len(request.images) != 2:
        return ToolResult(
            status="error",
            error="optical_sar_fusion requires exactly 2 images: one optical/multispectral and one SAR.",
        )

    modalities = [image.modality.lower() for image in request.images]
    optical_count = sum(modality in _OPTICAL_MODALITIES for modality in modalities)
    sar_count = modalities.count("sar")
    if optical_count != 1 or sar_count != 1:
        return ToolResult(
            status="error",
            error=(
                "optical_sar_fusion requires exactly one optical or multispectral "
                f"image and one SAR image; received {modalities}."
            ),
        )

    from satquery.inference.api import run_optical_sar_fusion

    optical_image = next(
        image
        for image in request.images
        if image.modality.lower() in _OPTICAL_MODALITIES
    )
    sar_image = next(image for image in request.images if image.modality.lower() == "sar")
    result = run_optical_sar_fusion(
        optical_image.metadata["array"], sar_image.metadata["array"]
    )
    return ToolResult(status="success", data={"inference_result": result})
