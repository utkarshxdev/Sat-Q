"""
satquery/inference/agent_registry.py
──────────────────────────────────────
Agent Tool Registry — for the Agentic Orchestrator (LLM tool-calling router).

The Architect's LLM router (Groq/Llama-3 or Gemini) uses JSON function-calling.
This module exposes every ML tool in a format that can be directly registered
as a tool call in any LangChain / LlamaIndex / raw OpenAI-tools-format router.

Usage:
    from satquery.inference.agent_registry import TOOL_REGISTRY, dispatch_tool

    # Get all tools for the LLM prompt
    tools = [t["schema"] for t in TOOL_REGISTRY.values()]

    # Execute a tool by name (called from LLM tool-call response)
    result = dispatch_tool("get_change_analysis", {
        "img_t1": ...,   # np.ndarray
        "img_t2": ...,
    })

Each tool entry in TOOL_REGISTRY contains:
    "fn"     : Callable — the actual Python function.
    "schema" : dict     — OpenAI-compatible function schema for LLM consumption.
    "input_spec" : dict — Runtime input validation spec.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

# ─── Import ML tools ─────────────────────────────────────────────────────────
from satquery.inference.api import get_change_analysis, extract_fused_features


# ─── JSON Schemas (OpenAI function-calling format) ──────────────────────────

_CHANGE_ANALYSIS_SCHEMA = {
    "name": "get_change_analysis",
    "description": (
        "Detects and quantifies land-cover changes between two co-registered "
        "bi-temporal satellite images. Returns a binary spatial change mask "
        "(uint8), a change-detected boolean, confidence score, and a human-readable "
        "summary. Use when the query involves 'what changed', 'before/after', "
        "'temporal change', or 'change detection'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "img_t1": {
                "type": "array",
                "description": (
                    "Pre-change image as a nested float32 array of shape (C, H, W) "
                    "with values in [0, 1]. Pass as list-of-lists or supply the key "
                    "'img_t1_key' to fetch from the session image store."
                ),
            },
            "img_t2": {
                "type": "array",
                "description": (
                    "Post-change image as a nested float32 array of shape (C, H, W) "
                    "with values in [0, 1]. Must share C, H, W with img_t1."
                ),
            },
            "threshold": {
                "type": "number",
                "description": "Change probability threshold (default 0.5).",
                "default": 0.5,
            },
        },
        "required": ["img_t1", "img_t2"],
    },
}

_FUSED_FEATURES_SCHEMA = {
    "name": "extract_fused_features",
    "description": (
        "Fuses optical (RGB/multispectral) and SAR (VV/VH) imagery via cross-attention "
        "to produce a joint 768-dimensional embedding and BigEarthNet-19 land-cover "
        "probability scores. Use when the query involves 'optical and SAR', "
        "'built-up detection', 'water body mapping', 'cross-modal analysis', "
        "or 'land cover classification'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "optical": {
                "type": "array",
                "description": "Optical image float32 (3, H, W) in [0, 1].",
            },
            "sar": {
                "type": "array",
                "description": "SAR image float32 (2, H, W) in [0, 1]. Channels: VV, VH.",
            },
        },
        "required": ["optical", "sar"],
    },
}


# ─── Runtime input converters ─────────────────────────────────────────────────

def _to_numpy(x: Any) -> np.ndarray:
    """Convert list/list-of-lists or np.ndarray to float32 np.ndarray."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.array(x, dtype=np.float32)


def _run_change_analysis(kwargs: dict) -> dict:
    img_t1    = _to_numpy(kwargs["img_t1"])
    img_t2    = _to_numpy(kwargs["img_t2"])
    threshold = float(kwargs.get("threshold", 0.5))
    result    = get_change_analysis(img_t1, img_t2, threshold=threshold)
    # Serialise change_mask for JSON transport (list of lists)
    result["change_mask"] = result["change_mask"].tolist()
    return result


def _run_fused_features(kwargs: dict) -> dict:
    optical = _to_numpy(kwargs["optical"])
    sar     = _to_numpy(kwargs["sar"])
    result  = extract_fused_features(optical, sar)
    # Serialise numpy arrays for JSON transport
    result["fused_embedding"]   = result["fused_embedding"].tolist()
    result["land_cover_logits"] = result["land_cover_logits"].tolist()
    result["land_cover_probs"]  = result["land_cover_probs"].tolist()
    result["attention_map"]     = result["attention_map"].tolist()
    return result


# ─── Registry ────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {
    "get_change_analysis": {
        "fn":         get_change_analysis,      # direct numpy call
        "runner":     _run_change_analysis,     # JSON-safe call
        "schema":     _CHANGE_ANALYSIS_SCHEMA,
        "input_spec": {
            "img_t1": "np.ndarray (C, H, W) float32",
            "img_t2": "np.ndarray (C, H, W) float32",
        },
        "output_spec": {
            "change_mask":       "np.ndarray uint8 (H, W) {0, 255}",
            "change_detected":   "bool",
            "confidence":        "float [0, 1]",
            "changed_area_pct":  "float",
            "summary":           "str",
            "model_used":        "str",
            "checkpoint_loaded": "bool",
        },
        "task_tags": ["change_detection", "bitemporal", "change_vqa"],
    },
    "extract_fused_features": {
        "fn":         extract_fused_features,   # direct numpy call
        "runner":     _run_fused_features,       # JSON-safe call
        "schema":     _FUSED_FEATURES_SCHEMA,
        "input_spec": {
            "optical": "np.ndarray (3, H, W) float32",
            "sar":     "np.ndarray (2, H, W) float32",
        },
        "output_spec": {
            "fused_embedding":   "np.ndarray float32 (768,)",
            "land_cover_logits": "np.ndarray float32 (19,)",
            "land_cover_probs":  "np.ndarray float32 (19,)",
            "attention_map":     "np.ndarray float32 (H_p, W_p)",
            "status":            "str",
            "model_used":        "str",
            "checkpoint_loaded": "bool",
        },
        "task_tags": ["optical_sar_fusion", "land_cover", "cross_modal"],
    },
}


def dispatch_tool(tool_name: str, kwargs: dict) -> dict:
    """
    Execute a registered tool by name.

    Called by the Agentic Orchestrator when the LLM selects a tool.

    Args:
        tool_name : One of the keys in TOOL_REGISTRY.
        kwargs    : Tool arguments (arrays may be lists-of-lists for JSON compat).

    Returns:
        JSON-serialisable result dict.

    Raises:
        KeyError  : If tool_name is not in TOOL_REGISTRY.
        ValueError: If required arguments are missing.
    """
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(
            f"Unknown tool: '{tool_name}'. "
            f"Available tools: {list(TOOL_REGISTRY.keys())}"
        )

    entry = TOOL_REGISTRY[tool_name]

    # Check required parameters against schema
    required = entry["schema"]["parameters"].get("required", [])
    missing  = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(
            f"Tool '{tool_name}' missing required arguments: {missing}"
        )

    return entry["runner"](kwargs)


def get_tool_schemas() -> list[dict]:
    """Return all tool schemas in OpenAI function-calling format."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]
