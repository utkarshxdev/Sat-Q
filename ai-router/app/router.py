"""Provider-independent query routing for the first SatQuery milestone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.registry import get_registered_tool
from app.schemas import RoutingDecision, ToolName, ToolRequest
from app.validator import validate_tool_request


class RoutingError(ValueError):
    """Raised when no registered tool can safely handle a request."""


class QueryRouter(Protocol):
    """Contract that deterministic and future LLM-backed routers share."""

    def route(self, request: ToolRequest) -> RoutingDecision:
        """Return a validated routing decision for a request."""


@dataclass(frozen=True)
class DeterministicKeywordRouter:
    """Routes well-defined first-milestone queries without an LLM provider."""

    def route(self, request: ToolRequest) -> RoutingDecision:
        """Select one compatible registered tool from query and image context."""
        selected_tool = self._select_candidate(request)
        self._validate_candidate(request, selected_tool)

        # The routing contract intentionally permits only these two fields.
        return RoutingDecision(selected_tool=selected_tool, parameters={})

    def _select_candidate(self, request: ToolRequest) -> ToolName:
        query = request.query.lower()
        modalities = {image.modality.lower() for image in request.images}

        if _has_change_intent(query):
            return "run_change_detection"

        if _has_fusion_intent(query):
            return "run_optical_sar_fusion"

        if _has_vqa_intent(query) and len(request.images) == 1:
            return "single_image_vqa"

        if len(request.images) == 1 and modalities <= {"optical", "multispectral", "sar"}:
            return "single_image_vqa"

        raise RoutingError(
            "Unable to route the request: the query and image context do not "
            "identify a compatible registered tool."
        )

    @staticmethod
    def _validate_candidate(request: ToolRequest, selected_tool: ToolName) -> None:
        # Resolving through the registry prevents arbitrary selected tool names.
        get_registered_tool(selected_tool)
        validation = validate_tool_request(request, selected_tool)
        if not validation.is_valid:
            raise RoutingError(
                f"Cannot route to '{selected_tool}': " + "; ".join(validation.errors)
            )



import os
import json

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@dataclass(frozen=True)
class GeminiLLMRouter:
    """Routes queries using Gemini 1.5 Flash for intelligent agentic orchestration."""

    def route(self, request: ToolRequest) -> RoutingDecision:
        if not GEMINI_AVAILABLE:
            raise RoutingError("google-generativeai or dotenv not installed.")
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        
        prompt = f"""
        You are the Agentic Controller for an ISRO remote sensing platform.
        Analyze the user's query and select the appropriate specialist tool.
        
        User Query: "{request.query}"
        Number of images provided: {len(request.images)}
        
        Allowed Tools:
        1. "single_image_vqa" - For describing or answering questions about ONE image.
        2. "run_change_detection" - For finding changes between TWO images (T1 and T2).
        3. "run_optical_sar_fusion" - For combining TWO images (one optical, one radar).
        
        Return ONLY a valid JSON object matching this schema:
        {{"selected_tool": "name_of_tool", "parameters": {{}}}}
        """
        
        try:
            response = model.generate_content(prompt)
            decision_dict = json.loads(response.text)
            selected_tool = decision_dict["selected_tool"]
        except Exception as e:
            raise RoutingError(f"Gemini routing failed: {str(e)}")
            
        # Ensure it's valid in the registry
        DeterministicKeywordRouter._validate_candidate(request, selected_tool)
        
        return RoutingDecision(
            selected_tool=selected_tool,
            parameters=decision_dict.get("parameters", {})
        )

def route_request(request: ToolRequest) -> RoutingDecision:
    """Route a request using Gemini if available, else fallback to deterministic."""
    if GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        return GeminiLLMRouter().route(request)
    return DeterministicKeywordRouter().route(request)



def _has_change_intent(query: str) -> bool:
    return (
        any(keyword in query for keyword in ("change", "changed", "difference"))
        or ("between" in query and ("date" in query or "time" in query))
    )


def _has_fusion_intent(query: str) -> bool:
    has_optical_reference = "optical" in query or "multispectral" in query
    has_sar_reference = "sar" in query or "radar" in query
    return "fusion" in query or "together" in query or (
        has_optical_reference and has_sar_reference
    )


def _has_vqa_intent(query: str) -> bool:
    return any(
        phrase in query
        for phrase in ("describe", "what is", "identify", "where is", "how many")
    )
