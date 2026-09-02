"""
app.py
──────
SatQuery AI — Interactive Vision-Language Assistant GUI.

Features:
  1. Bi-Temporal Change Analysis (T1 vs T2 with spatial change overlay & metrics)
  2. Optical-SAR Multimodal Fusion (Cross-attention heatmap & land cover classification)
  3. Agentic NL Query Assistant (Query interpretation, tool dispatch & auditable trace)
  4. Geospatial Report Generator (Downloadable audit summary)

Launch locally:
    streamlit run app.py
"""
from __future__ import annotations

import json
import time
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

from satquery.inference.api import get_change_analysis, extract_fused_features
from satquery.inference.agent_registry import dispatch_tool, TOOL_REGISTRY, get_tool_schemas


# ─── Page Config ─────────────────────────────────────────────────────────────
if _STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="SatQuery AI — Multimodal Remote Sensing Assistant",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ─── Helper Functions ────────────────────────────────────────────────────────

def _load_image_to_numpy(uploaded_file, target_channels: int = 3, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Converts uploaded file into float32 (C, H, W) in range [0, 1]."""
    img = Image.open(uploaded_file).convert("RGB" if target_channels == 3 else "L")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    if target_channels == 3:
        # (H, W, 3) -> (3, H, W)
        return arr.transpose(2, 0, 1)
    elif target_channels == 2:
        # SAR VV/VH mock from grayscale or 2 channels
        return np.stack([arr, np.roll(arr, 5, axis=0)], axis=0)
    else:
        return arr[np.newaxis, ...]


def _generate_synthetic_sample(scenario: str) -> dict:
    """Generates synthetic imagery for quick zero-upload demonstration."""
    rng = np.random.default_rng(int(time.time()) % 1000)
    h, w = 256, 256

    if scenario == "Urban Expansion (Bi-temporal)":
        t1 = np.ones((3, h, w), dtype=np.float32) * 0.3
        t1[1, :, :] += 0.2  # greenish / vegetation
        t2 = t1.copy()
        # Create built-up cluster in bottom right
        t2[:, 120:220, 120:220] = 0.85
        return {"t1": t1, "t2": t2, "name": "Urban Expansion"}

    elif scenario == "Flood Inundation (Bi-temporal)":
        t1 = np.ones((3, h, w), dtype=np.float32) * 0.45
        t2 = t1.copy()
        # Flood water channel (dark blue)
        t2[0, 60:180, :] = 0.05
        t2[1, 60:180, :] = 0.15
        t2[2, 60:180, :] = 0.55
        return {"t1": t1, "t2": t2, "name": "Flood Inundation"}

    elif scenario == "Optical-SAR Coastal / Port":
        opt = np.zeros((3, h, w), dtype=np.float32)
        opt[2, :, :128] = 0.6  # water on left
        opt[1, :, 128:] = 0.5  # land on right
        sar = np.zeros((2, h, w), dtype=np.float32)
        sar[0, :, :128] = 0.05 # low water backscatter
        sar[0, :, 128:] = 0.7  # high urban backscatter
        # Ship / structure in water (high SAR return)
        sar[:, 100:130, 40:70] = 0.95
        return {"optical": opt, "sar": sar, "name": "Coastal & Port"}

    return {}


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    if not _STREAMLIT_AVAILABLE:
        print("Streamlit not installed in this environment. Run: pip install streamlit")
        return

    st.title("🛰️ SatQuery AI — Vision-Language Remote Sensing Assistant")
    st.markdown(
        "**Problem Statement 26167 (ISRO/SAC)** | Multimodal Optical-SAR Fusion & Bi-Temporal Change Detection Engine"
    )

    sidebar = st.sidebar
    sidebar.header("Navigation & Settings")
    mode = sidebar.radio(
        "Select Operation Mode:",
        [
            "🔄 Bi-Temporal Change Detection",
            "🔬 Optical-SAR Multimodal Fusion",
            "🤖 Agentic Query & Tool Orchestrator",
            "📋 Architecture & System Audit",
        ],
    )

    sidebar.markdown("---")
    sidebar.subheader("Hardware & Execution")
    import torch
    from satquery.config import DEVICE
    sidebar.info(f"Active Hardware: **{DEVICE}**")
    sidebar.markdown(f"- PyTorch: `{torch.__version__}`")
    sidebar.markdown(f"- Model Core: `SeCo ResNet34 + ViT/ConvNeXt`")

    # ── TAB 1: Bi-Temporal Change Detection ──────────────────────────────────
    if mode == "🔄 Bi-Temporal Change Detection":
        st.header("Bi-Temporal Change Detection (Siamese U-Net)")
        st.caption("Identify and quantify semantic changes between spatially co-registered multi-temporal observations.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image T1 (Pre-Change)")
            f1 = st.file_uploader("Upload T1 (GeoTIFF/PNG/JPG)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="t1")
        with col2:
            st.subheader("Image T2 (Post-Change)")
            f2 = st.file_uploader("Upload T2 (GeoTIFF/PNG/JPG)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="t2")

        preset = st.selectbox(
            "Or load a pre-configured scenario:",
            ["None", "Urban Expansion (Bi-temporal)", "Flood Inundation (Bi-temporal)"]
        )

        threshold = st.slider("Change Probability Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        if st.button("🚀 Run Change Analysis", type="primary"):
            if f1 and f2:
                img_t1 = _load_image_to_numpy(f1, target_channels=3)
                img_t2 = _load_image_to_numpy(f2, target_channels=3)
            elif preset != "None":
                sample = _generate_synthetic_sample(preset)
                img_t1 = sample["t1"]
                img_t2 = sample["t2"]
            else:
                st.warning("Please upload two images or select a preset scenario.")
                return

            with st.spinner("Executing Siamese U-Net difference pipeline..."):
                t0 = time.perf_counter()
                result = get_change_analysis(img_t1, img_t2, threshold=threshold)
                elapsed = (time.perf_counter() - t0) * 1000

            st.success(f"Analysis completed in {elapsed:.1f} ms on {DEVICE}")

            # Display Visual Evidence
            st.subheader("Visual Evidence & Grounding")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            t1_display = (img_t1.transpose(1, 2, 0) * 255).astype(np.uint8)
            t2_display = (img_t2.transpose(1, 2, 0) * 255).astype(np.uint8)
            mask_display = result["change_mask"]

            # Overlay: highlight change in red on T2
            overlay = t2_display.copy()
            overlay[mask_display == 255] = [255, 30, 30]

            with res_col1:
                st.image(t1_display, caption="Image T1 (Pre-Change)", use_container_width=True)
            with res_col2:
                st.image(t2_display, caption="Image T2 (Post-Change)", use_container_width=True)
            with res_col3:
                st.image(mask_display, caption="Binary Change Mask (255=Change)", use_container_width=True)
            with res_col4:
                st.image(overlay, caption="Spatial Change Overlay", use_container_width=True)

            # Quantitative Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Change Detected", "YES" if result["change_detected"] else "NO")
            mcol2.metric("Changed Area", f"{result['changed_area_pct']}%")
            mcol3.metric("Confidence Score", f"{result['confidence']:.3f}")
            mcol4.metric("Model Architecture", result["model_used"])

            st.info(f"**Natural Language Summary:** {result['summary']}")

    # ── TAB 2: Optical-SAR Multimodal Fusion ─────────────────────────────────
    elif mode == "🔬 Optical-SAR Multimodal Fusion":
        st.header("Optical-SAR Cross-Attention Fusion")
        st.caption("Joint representation extraction combining optical spectral content with SAR structural backscatter.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optical / Multispectral Image")
            f_opt = st.file_uploader("Upload Optical (RGB)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="opt")
        with col2:
            st.subheader("SAR Co-Registered Image")
            f_sar = st.file_uploader("Upload SAR (VV/VH)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="sar")

        preset_fuse = st.selectbox(
            "Or load a pre-configured cross-modal pair:",
            ["None", "Optical-SAR Coastal / Port"]
        )

        if st.button("🚀 Extract Fused Features & Classify", type="primary"):
            if f_opt and f_sar:
                opt_np = _load_image_to_numpy(f_opt, target_channels=3, size=(224, 224))
                sar_np = _load_image_to_numpy(f_sar, target_channels=2, size=(224, 224))
            elif preset_fuse != "None":
                sample = _generate_synthetic_sample(preset_fuse)
                opt_np = sample["optical"]
                sar_np = sample["sar"]
            else:
                st.warning("Please upload optical and SAR images or select a preset.")
                return

            with st.spinner("Fusing modalities across cross-attention bottleneck..."):
                t0 = time.perf_counter()
                result = extract_fused_features(opt_np, sar_np)
                elapsed = (time.perf_counter() - t0) * 1000

            st.success(f"Fusion extracted {result['fused_embedding'].shape[0]}-D joint embedding in {elapsed:.1f} ms")

            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.image((opt_np.transpose(1, 2, 0) * 255).astype(np.uint8), caption="Optical (Spectral)", use_container_width=True)
            with vcol2:
                # Composite SAR (VV=red, VH=green, ratio=blue)
                sar_vis = np.zeros((224, 224, 3), dtype=np.uint8)
                sar_vis[:, :, 0] = (sar_np[0] * 255).astype(np.uint8)
                sar_vis[:, :, 1] = (sar_np[1] * 255).astype(np.uint8)
                sar_vis[:, :, 2] = ((sar_np[0] / (sar_np[1] + 1e-4)) * 128).clip(0, 255).astype(np.uint8)
                st.image(sar_vis, caption="SAR Co-Registered (Structural)", use_container_width=True)
            with vcol3:
                attn = result["attention_map"]
                if attn.ndim == 2:
                    attn_norm = ((attn - attn.min()) / (attn.max() - attn.min() + 1e-8) * 255).astype(np.uint8)
                    st.image(attn_norm, caption="Optical-SAR Cross-Attention Heatmap", use_container_width=True)

            # Top Land Cover Predictions (BigEarthNet-19)
            st.subheader("BigEarthNet-19 Land Cover Classification")
            labels = [
                "Urban fabric", "Industrial/commercial", "Arable land", "Permanent crops",
                "Pastures", "Complex cultivation", "Land principally agriculture",
                "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland",
                "Moors & heathland", "Sclerophyllous vegetation", "Transitional woodland",
                "Beaches/dunes/sands", "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"
            ]
            probs = result["land_cover_probs"]
            top_indices = np.argsort(probs)[::-1][:5]

            for idx in top_indices:
                st.write(f"**{labels[idx]}**")
                st.progress(float(probs[idx]), text=f"{float(probs[idx])*100:.1f}%")

    # ── TAB 3: Agentic NL Query Assistant ────────────────────────────────────
    elif mode == "🤖 Agentic Query & Tool Orchestrator":
        st.header("Agentic Vision-Language Query Assistant")
        st.caption("Natural language query parsing, automated tool dispatch, and auditable execution trace.")

        user_query = st.text_input(
            "Enter Natural Language Remote Sensing Query:",
            value="What changed between these two dates, and where did the change occur?",
        )

        st.markdown("**Example Domain Queries:**")
        ex1, ex2, ex3 = st.columns(3)
        if ex1.button("🔍 'What changed between these two dates?'"):
            user_query = "What changed between these two dates, and where did the change occur?"
        if ex2.button("🌿 'Use optical and SAR images to identify built-up areas.'"):
            user_query = "Use the optical and SAR images together to identify built-up and water-covered regions."
        if ex3.button("🏗️ 'Has the built-up area increased or decreased?'"):
            user_query = "Has the built-up area increased, decreased, or remained unchanged?"

        if st.button("⚡ Dispatch Agent Plan & Execute", type="primary"):
            st.subheader("Auditable Execution Trace")

            # Simulate LLM Tool Router Classification
            q_lower = user_query.lower()
            if "change" in q_lower or "dates" in q_lower or "increased" in q_lower or "decreased" in q_lower:
                selected_tool = "get_change_analysis"
                task_type = "Bi-Temporal Change Reasoning (Task 26167.3)"
            else:
                selected_tool = "extract_fused_features"
                task_type = "Optical-SAR Cross-Modal Fusion (Task 26167.4)"

            trace_col1, trace_col2 = st.columns(2)
            with trace_col1:
                st.markdown(f"**1. Query Intent:** `{task_type}`")
                st.markdown(f"**2. Selected Specialist Tool:** `{selected_tool}`")
                st.markdown(f"**3. Target Hardware:** `{DEVICE}`")

            with trace_col2:
                st.markdown("**4. Tool Schema Parameters:**")
                st.code(json.dumps(TOOL_REGISTRY[selected_tool]["schema"], indent=2), language="json")

            with st.spinner("Executing routed specialist tool..."):
                if selected_tool == "get_change_analysis":
                    sample = _generate_synthetic_sample("Urban Expansion (Bi-temporal)")
                    res = get_change_analysis(sample["t1"], sample["t2"])
                else:
                    sample = _generate_synthetic_sample("Optical-SAR Coastal / Port")
                    res = extract_fused_features(sample["optical"], sample["sar"])

            st.success("Tool execution finished with valid contract response.")
            st.json({k: str(v) if isinstance(v, np.ndarray) else v for k, v in res.items()})

    # ── TAB 4: Architecture & System Audit ───────────────────────────────────
    elif mode == "📋 Architecture & System Audit":
        st.header("SatQuery AI System Architecture")
        st.markdown("""
        ### Core Pipeline Highlights
        1. **Dual Encoders**: ViT-B/16 (Optical) + ConvNeXt-Tiny (SAR adapted for 2-channel VV/VH).
        2. **Cross-Attention Bottleneck**: Optical queries attend to SAR spatial keys/values to ground spectral signatures with structural radar backscatter.
        3. **True Siamese U-Net**: Shared ResNet34 backbone computing multi-scale absolute difference maps \(|F_{T1} - F_{T2}|\) across 4 skip levels.
        4. **Compound Loss**: FocalLoss (\(\gamma=2, \alpha=0.75\)) + Soft Dice Loss for extreme class-imbalance (<5% change pixels).
        5. **Empty-Mask Defense**: Zeroes out false positives below the 0.5% spatial noise floor.
        6. **ONNX & MPS Acceleration**: Optimized sub-300ms latency execution for live demonstrations.
        """)

        st.subheader("Available Agent Tool Registry")
        st.json(get_tool_schemas())


if __name__ == "__main__":
    main()
