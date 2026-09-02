# SatQuery AI
**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis**
**Problem Statement:** ISRO / SAC PS 26167

---

## 1. Executive Summary
**SatQuery AI** is an agentic, query-driven vision-language platform designed to democratize complex remote sensing analysis. Traditionally, analyzing satellite data requires deep GIS expertise, manual model selection, and domain-specific knowledge. SatQuery AI eliminates this barrier by allowing users to upload multimodal satellite imagery (Optical/Multispectral and SAR) and query it using natural language. 

Instead of relying on a single monolithic Vision-Language Model (VLM) prone to hallucinations, SatQuery AI employs an **Agentic Orchestration Framework**. A high-speed Large Language Model (LLM) acts as a router, interpreting user intent, validating inputs, and dynamically invoking specialized PyTorch vision models. The system extracts evidence-grounded insights, combining structural radar data with spectral optical data, and presents them in an interactive, ISRO-themed Command Center GUI.

---

## 2. The Agentic Architecture
The core innovation of SatQuery AI is its modular, intent-driven backend.

1. **Query & Ingestion:** The user uploads GeoTIFF/PNG imagery (e.g., Cartosat-2S, RISAT, Sentinel) and submits a text query (e.g., *"What changed between these two dates?"*).
2. **The Architect (LLM Router):** Powered by Gemini 1.5 Flash (via `google-genai`), the routing agent analyzes the query. It validates the number of images, their modalities, and outputs a strict JSON `RoutingDecision`.
3. **Tool Registry:** The decision is validated against strict schemas (e.g., preventing a single image from triggering a bi-temporal change detection tool).
4. **Specialist Execution:** The system routes the tensors to the appropriate highly optimized, fine-tuned `.safetensors` model.
5. **Auditable Response:** The outputs (heatmaps, bounding boxes, text answers) are aggregated into a `FinalResponse` object containing an execution trace, ensuring 100% transparency of the AI's decision-making process.

---

## 3. Core Specialist Modules

### Module A: Optical-SAR Fusion
* **Purpose:** Joint reasoning over co-registered Optical and Synthetic Aperture Radar (SAR) pairs.
* **Architecture:** Uses a Cross-Attention mechanism (merging ViT and ConvNeXt backbones) to extract complementary information. Optical provides spectral/contextual data, while SAR penetrates clouds and provides structural/texture data.
* **Output:** Land-cover classification (BigEarthNet-19 schema) and fused feature heatmaps.

### Module B: Bi-Temporal Change Detection
* **Purpose:** Identifying spatial and structural changes between two dates ($T_1$ and $T_2$).
* **Architecture:** A Siamese U-Net backbone processing spatially corresponding image pairs.
* **Output:** A precise binary spatial change map, total area changed percentage, and confidence metrics.

### Module C: Visual Question Answering (VQA) & Grounding
* **Purpose:** Single-image understanding and text-guided region bounding.
* **Architecture:** A specialized VLM (Qwen2-VL / PaliGemma) fine-tuned with QLoRA specifically on remote sensing datasets.
* **Output:** Natural language answers and bounding box coordinates citing exact geographic regions corresponding to the text.

---

## 4. Geospatial Data Pipeline
To ensure compatibility with ISRO's evaluation criteria (hidden Cartosat-2S and RISAT SAR datasets), the system includes a robust preprocessing engine using `rasterio`:
* **GeoTIFF Parsing:** Reads 16-bit geospatial TIFFs natively.
* **Percentile Normalization:** Clips extreme pixel anomalies (2nd to 98th percentile) to convert raw satellite reflectance data into standard normalized `(C, H, W)` PyTorch tensors.

---

## 5. Training & Infrastructure
* **Supercomputer Scale:** The underlying specialist models are trained on an **NVIDIA DGX H200 Supercomputer** (8× H200 GPUs, 141 GB VRAM) utilizing PyTorch Distributed Data Parallel (DDP) and Mixed Precision (FP16).
* **Datasets:** Fine-tuned primarily on **BigEarthNet** (~590,000 multimodal image pairs), along with VRSBench, RSVQA, and CDVQA for text/captioning alignment.
* **Efficient Inference:** Final models are compiled into lightweight `.safetensors` to minimize optimizer bloat and ensure fast loading in production environments.

---

## 6. The User Interface (Command Center)
The frontend is a custom-built Streamlit application styled as a professional **ISRO Command Center**. 
* **Aesthetic:** Deep Space Blue (`#04142C`) and ISRO Saffron (`#FF671F`) color palette with strict industrial UI components.
* **Geospatial Workspace:** Center pane featuring a dual-swipe $T_1 \leftrightarrow T_2$ slider, layer stack opacity toggles, and interactive map overlays.
* **Architect Audit Log:** A dedicated pane displaying the exact JSON execution trace of the LLM router, ensuring judges can audit tool selection, latencies, and validation checks.

---

## 7. ISRO Compliance Checklist
SatQuery AI directly satisfies all mandatory functional scopes defined in PS 26167:
- [x] **Remote-Sensing Adaptation:** Models fine-tuned on BigEarthNet.txt.
- [x] **Single-Image Baseline:** VQA and region grounding implemented.
- [x] **Multi-Image Change Analysis:** Bi-temporal change mapping and metrics.
- [x] **Cross-Modal Pair Analysis:** Optical+SAR joint fusion network.
- [x] **Agentic Orchestration:** Automated tool selection with auditable execution trace.
