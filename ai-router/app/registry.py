"""Registry of specialist tools approved for SatQuery orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas import ToolName, ToolRequest, ToolResult
from app.tools.vqa import single_image_vqa
from app.tools.change import run_change_detection_tool
from app.tools.optical_sar import run_optical_sar_fusion_tool


ToolImplementation = Callable[[ToolRequest], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and callable contract for one allowed specialist tool."""

    name: ToolName
    description: str
    required_image_count: int
    supported_configurations: tuple[str, ...]
    supported_modalities: tuple[str, ...]
    parameters_schema: dict[str, Any]
    implementation: ToolImplementation
    tool_identifier: str


TOOL_REGISTRY: dict[ToolName, ToolDefinition] = {
    "single_image_vqa": ToolDefinition(
        name="single_image_vqa",
        description="Answers a natural-language question about one remote-sensing image.",
        required_image_count=1,
        supported_configurations=("single_image",),
        supported_modalities=("optical", "multispectral", "sar"),
        parameters_schema={},
        implementation=single_image_vqa,
        tool_identifier="mock-single-image-vqa-v1",
    ),
    "run_change_detection": ToolDefinition(
        name="run_change_detection",
        description="Detects and describes changes between a bi-temporal image pair.",
        required_image_count=2,
        supported_configurations=("bi_temporal",),
        supported_modalities=("optical", "multispectral", "sar"),
        parameters_schema={
            "image_configuration": {
                "type": "string",
                "allowed_values": ("bi_temporal",),
                "required": False,
            }
        },
        implementation=run_change_detection_tool,
        tool_identifier="satquery.inference.api.run_change_detection",
    ),
    "run_optical_sar_fusion": ToolDefinition(
        name="run_optical_sar_fusion",
        description="Combines one optical or multispectral image with one SAR image.",
        required_image_count=2,
        supported_configurations=("optical_sar_pair",),
        supported_modalities=("optical", "multispectral", "sar"),
        parameters_schema={},
        implementation=run_optical_sar_fusion_tool,
        tool_identifier="satquery.inference.api.run_optical_sar_fusion",
    ),
}


def get_registered_tool(tool_name: str) -> ToolDefinition:
    """Return an approved tool definition or raise a clear error for unknown tools."""
    try:
        return TOOL_REGISTRY[tool_name]  # type: ignore[index]
    except KeyError as error:
        allowed_tools = ", ".join(TOOL_REGISTRY)
        raise ValueError(
            f"Unknown tool '{tool_name}'. Allowed tools are: {allowed_tools}."
        ) from error
