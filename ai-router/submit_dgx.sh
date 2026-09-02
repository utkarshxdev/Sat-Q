#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# submit_dgx.sh — SLURM job script for NVIDIA DGX H200
# ═══════════════════════════════════════════════════════════════════════════
#
# Submit:  sbatch submit_dgx.sh
# Monitor: squeue -u $USER
# Cancel:  scancel <job_id>
# Logs:    tail -f slurm-<job_id>.out
#
# ═══════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=satquery-train
#SBATCH --partition=gpu                # adjust to your cluster's GPU partition
#SBATCH --nodes=1                      # single DGX node
#SBATCH --ntasks-per-node=1            # 1 task, torchrun handles GPU processes
#SBATCH --gpus-per-node=8              # all 8 H200 GPUs
#SBATCH --cpus-per-task=112            # all 112 CPU cores on DGX H200
#SBATCH --mem=0                        # use all available RAM (~2 TB)
#SBATCH --time=12:00:00                # 12 hours wall time
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ─── Environment ─────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "SatQuery AI — DGX H200 Training"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $SLURM_GPUS_ON_NODE"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Started:   $(date)"
echo "═══════════════════════════════════════════════════════════"

# Load modules (adjust to your cluster's module system)
# module load cuda/12.4
# module load python/3.11
# module load nccl

# Activate virtual environment (adjust path)
# source /path/to/your/venv/bin/activate

# Project root
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# NCCL optimisations for DGX H200
export NCCL_IB_DISABLE=0              # enable InfiniBand
export NCCL_NET_GDR_LEVEL=5           # GPUDirect RDMA
export NCCL_P2P_LEVEL=NVL             # NVLink for intra-node
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=14             # 112 cores / 8 GPUs
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Verify GPU setup
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.device_count()} GPUs')"

# Ensure configilm is installed (required for BigEarthNet.txt LMDB loading)
pip install -q configilm 2>/dev/null || echo "⚠ configilm install failed — bentxt mode may not work"

# ─── DATA PATHS ──────────────────────────────────────────────────────────
# ⚠️  UPDATE THESE to match your cluster's filesystem:
LEVIR_CD_ROOT="/data/LEVIR-CD"                       # LEVIR-CD dataset
BIGEARTHNET_LMDB="/data/BigEarthNet/BENv2.lmdb"     # LMDB encoded images
BIGEARTHNET_PARQUET="/data/BigEarthNet"               # metadata.parquet dir
BIGEARTHNET_DISK="/data/BigEarthNet"                  # Fallback: raw images

echo ""
echo "LEVIR-CD:       $LEVIR_CD_ROOT"
echo "BEN LMDB:       $BIGEARTHNET_LMDB"
echo "BEN Parquet:    $BIGEARTHNET_PARQUET"

# ─── PHASE 1: Change Detection (Siamese U-Net) ──────────────────────────
echo ""
echo "═══ PHASE 1: Change Detection Training ═══"
echo "  8× H200 | batch=32/GPU | effective=256 | 100 epochs | img=512"
echo ""

torchrun \
    --standalone \
    --nproc_per_node=8 \
    satquery/training/train_change_ddp.py \
        --data_root "$LEVIR_CD_ROOT" \
        --epochs 100 \
        --batch_size 32 \
        --img_size 512

CHANGE_EXIT=$?
echo "Change Detection exit code: $CHANGE_EXIT"

# ─── PHASE 2: Optical-SAR Fusion (BigEarthNet.txt) ──────────────────────
echo ""
echo "═══ PHASE 2: Optical-SAR Fusion Training ═══"

# Priority A: BigEarthNet.txt LMDB (MANDATED — real Sentinel-1 SAR)
if [ -d "$BIGEARTHNET_LMDB" ]; then
    echo "  ★ Using BigEarthNet.txt with REAL Sentinel-1 SAR (LMDB)"
    echo "  8× H200 | batch=64/GPU | effective=512 | 30 epochs"
    echo ""
    torchrun \
        --standalone \
        --nproc_per_node=8 \
        satquery/training/train_fusion_ddp.py \
            --data_source bentxt \
            --lmdb_dir "$BIGEARTHNET_LMDB" \
            --parquet_dir "$BIGEARTHNET_PARQUET" \
            --epochs 30 \
            --batch_size 64

# Priority B: Raw images on disk (synthetic SAR)
elif [ -d "$BIGEARTHNET_DISK/train" ]; then
    echo "  ⚠ LMDB not found — using disk images (SAR will be synthetic)"
    echo ""
    torchrun \
        --standalone \
        --nproc_per_node=8 \
        satquery/training/train_fusion_ddp.py \
            --data_source disk \
            --data_root "$BIGEARTHNET_DISK" \
            --epochs 30 \
            --batch_size 64

# Priority C: Stream from HuggingFace (synthetic SAR)
else
    echo "  ⚠ No local data — streaming 200k samples from HuggingFace"
    echo ""
    torchrun \
        --standalone \
        --nproc_per_node=8 \
        satquery/training/train_fusion_ddp.py \
            --data_source hf \
            --max_samples 200000 \
            --epochs 20 \
            --batch_size 64
fi

FUSION_EXIT=$?
echo "Fusion exit code: $FUSION_EXIT"

# ─── PHASE 3: ONNX Export ───────────────────────────────────────────────
echo ""
echo "═══ PHASE 3: ONNX Export ═══"
python satquery/inference/onnx_export.py
echo "ONNX export exit code: $?"

# ─── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Training Complete"
echo "Finished: $(date)"
echo ""
echo "Checkpoints:"
ls -lh checkpoints/ 2>/dev/null
echo ""
echo "ONNX models:"
ls -lh onnx_models/ 2>/dev/null
echo ""

if [ $CHANGE_EXIT -eq 0 ] && [ $FUSION_EXIT -eq 0 ]; then
    echo "✓ ALL PHASES PASSED"
else
    echo "✗ SOME PHASES FAILED (change=$CHANGE_EXIT, fusion=$FUSION_EXIT)"
fi
echo "═══════════════════════════════════════════════════════════"
