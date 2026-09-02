"""Mock implementation of the single-image visual question answering tool."""

from app.schemas import ToolRequest, ToolResult


_SUPPORTED_MODALITIES = {"optical", "multispectral", "sar"}


def single_image_vqa(request: ToolRequest) -> ToolResult:
    """Return a deterministic placeholder VQA result for one supported image."""
    if len(request.images) != 1:
        return ToolResult(
            status="error",
            error="single_image_vqa requires exactly 1 image.",
        )

    image = request.images[0]
    modality = image.modality.lower()
    if modality not in _SUPPORTED_MODALITIES:
        return ToolResult(
            status="error",
            error=(
                "single_image_vqa supports optical, multispectral, or SAR "
                f"images; received '{image.modality}'."
            ),
        )

    return ToolResult(
        status="success",
        answer=(
            "Mock VQA result: this placeholder analysis received one "
            f"{modality} image for the supplied query."
        ),
        confidence=0.75,
        evidence={
            "type": "mock_image_reference",
            "image_id": image.id,
            "label": "No real visual grounding is produced by the mock tool.",
        },
        data={
            "tool_identifier": "mock-single-image-vqa-v1",
            "image_modality": modality,
            "query": request.query,
        },
    )
