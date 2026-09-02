"""Input validation for tool requests before orchestration executes a tool."""

from __future__ import annotations

from dataclasses import dataclass

from app.registry import ToolDefinition, get_registered_tool
from app.schemas import ImageContext, ToolRequest


@dataclass(frozen=True)
class ValidationResult:
    """The observable outcome of request validation."""

    is_valid: bool
    errors: tuple[str, ...] = ()


def validate_tool_request(request: ToolRequest, tool_name: str) -> ValidationResult:
    """Validate a request against one registered tool without executing it."""
    try:
        tool = get_registered_tool(tool_name)
    except ValueError as error:
        return ValidationResult(is_valid=False, errors=(str(error),))

    errors: list[str] = []
    if not request.query.strip():
        errors.append("Query must not be empty.")

    _validate_image_count(request, tool, errors)
    _validate_modalities(request, tool, errors)

    if tool.name == "run_change_detection" and len(request.images) != 2:
        errors.append("Bi-temporal change detection requires exactly 2 images.")
    elif tool.name == "run_optical_sar_fusion":
        _validate_optical_sar_pair(request, errors)
    elif tool.name == "single_image_vqa" and len(request.images) != 1:
        errors.append("single_image_vqa requires exactly 1 supported image.")

    if tool.name in {"run_change_detection", "run_optical_sar_fusion"}:
        for image in request.images:
            if "array" not in image.metadata:
                errors.append(f"Image '{image.id}' is missing its array payload.")

    if tool.name == "run_change_detection" and len(request.images) == 2:
        roles = {
            str(image.metadata.get("temporal_role", "")).lower()
            for image in request.images
        }
        if request.parameters.get("image_configuration") != "bi_temporal" and roles != {
            "t1",
            "t2",
        }:
            errors.append("run_change_detection requires bi-temporal context.")

    return ValidationResult(is_valid=not errors, errors=tuple(errors))


def _validate_image_count(
    request: ToolRequest, tool: ToolDefinition, errors: list[str]
) -> None:
    if len(request.images) != tool.required_image_count:
        errors.append(
            f"{tool.name} requires exactly {tool.required_image_count} image(s); "
            f"received {len(request.images)}."
        )


def _validate_modalities(
    request: ToolRequest, tool: ToolDefinition, errors: list[str]
) -> None:
    supported_modalities = set(tool.supported_modalities)
    images: list[ImageContext] = request.images
    for image in images:
        modality = image.modality.lower()
        if modality not in supported_modalities:
            errors.append(
                f"Image '{image.id}' has unsupported modality '{image.modality}' "
                f"for {tool.name}. Supported modalities: "
                f"{', '.join(tool.supported_modalities)}."
            )


def _validate_optical_sar_pair(request: ToolRequest, errors: list[str]) -> None:
    images: list[ImageContext] = request.images
    modalities = [image.modality.lower() for image in images]
    optical_count = sum(modality in {"optical", "multispectral"} for modality in modalities)
    sar_count = modalities.count("sar")
    if optical_count != 1 or sar_count != 1:
        errors.append(
            "optical_sar_fusion requires exactly one optical or multispectral "
            f"image and one SAR image; received {modalities}."
        )
