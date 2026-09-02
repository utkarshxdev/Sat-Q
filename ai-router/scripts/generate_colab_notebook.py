"""
scripts/generate_colab_notebook.py
───────────────────────────────────
Generates a complete, executable Google Colab notebook (.ipynb)
with 1-click zip extraction, package install, and training cells.
"""
import json
from pathlib import Path

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛰️ SatQuery AI — Google Colab GPU Fine-Tuning Pipeline\n",
                "### Problem Statement 26167 (ISRO/SAC): Multimodal Optical-SAR Fusion & Bi-Temporal Change Detection\n",
                "\n",
                "#### How to start:\n",
                "1. Drag and drop **`satquery_colab_bundle.zip`** into the Colab Files tab on the left sidebar.\n",
                "2. Run the cells sequentially below."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Unzip project files into Colab\n",
                "!unzip -o satquery_colab_bundle.zip -d /content/\n",
                "%cd /content\n",
                "!ls -la /content/satquery"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Verify GPU availability\n",
                "!nvidia-smi\n",
                "import torch\n",
                "print('PyTorch CUDA available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU Device:', torch.cuda.get_device_name(0))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Install ML dependencies\n",
                "!pip install -q timm einops segmentation-models-pytorch datasets transformers huggingface-hub onnx onnxruntime-gpu"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 4: Train Optical-SAR Fusion Model (Streaming BigEarthNet from HF)\n",
                "!python satquery/training/train_fusion.py --epochs 5 --batch_size 16 --max_samples 10000"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Train Siamese Change Detection U-Net on LEVIR-CD\n",
                "!python satquery/training/train_change.py --epochs 30 --batch_size 16 --img_size 512"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 6: Export Checkpoints to ONNX Graphs\n",
                "!python satquery/inference/onnx_export.py --model all"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 7: Benchmark Latency (< 300 ms target)\n",
                "!python scripts/benchmark_latency.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 8: Zip checkpoints for downloading back to local Mac\n",
                "!zip -r trained_checkpoints.zip checkpoints/ onnx_models/\n",
                "from google.colab import files\n",
                "files.download('trained_checkpoints.zip')"
            ]
        }
    ]
}

out_dir = Path("notebooks")
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "Colab_Training_SatQuery_AI.ipynb"

with open(out_file, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Updated Colab notebook: {out_file}")
