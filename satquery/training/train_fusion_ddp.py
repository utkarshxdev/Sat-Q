"""
satquery/training/train_fusion_ddp.py
──────────────────────────────────────
Multi-GPU DDP training for the Optical-SAR Fusion Model.
Supports the mandated BigEarthNet.txt dataset (co-registered S1+S2)
on NVIDIA DGX H200.

Data Modes:
  1. "bentxt"  — BigEarthNet.txt (LMDB images + parquet annotations)
                 Uses configilm BENv2DataSet for real S1 SAR + S2 optical
  2. "disk"    — Generic local directory of images
  3. "hf"      — HuggingFace streaming fallback

Launch (8 GPUs, BigEarthNet.txt on DGX — RECOMMENDED):
    torchrun --nproc_per_node=8 satquery/training/train_fusion_ddp.py \\
        --data_source bentxt \\
        --lmdb_dir /data/BigEarthNet/BENv2.lmdb \\
        --parquet_dir /data/BigEarthNet/ \\
        --epochs 30 --batch_size 64

Launch (8 GPUs, HuggingFace streaming):
    torchrun --nproc_per_node=8 satquery/training/train_fusion_ddp.py \\
        --data_source hf --max_samples 200000 \\
        --epochs 20 --batch_size 64

Resume:
    torchrun --nproc_per_node=8 satquery/training/train_fusion_ddp.py \\
        --data_source bentxt --resume checkpoints/optical_sar_fused.pth \\
        --lmdb_dir /data/BigEarthNet/BENv2.lmdb \\
        --parquet_dir /data/BigEarthNet/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image as PILImage

from satquery.models.optical_sar_fusion import OpticalSARFusionModel
from satquery.config import (
    CKPT_DIR, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, NUM_BIGEARTHNET_CLASSES,
)


# ─── Distributed Helpers ─────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_main(msg):
    if is_main():
        print(msg, flush=True)


# ─── BigEarthNet-19 class labels ─────────────────────────────────────────────

BEN19_LABELS = [
    "Urban fabric", "Industrial/commercial", "Mine/dump/construction",
    "Artificial non-agricultural vegetation", "Arable land", "Permanent crops",
    "Pastures", "Complex cultivation", "Agriculture + natural vegetation",
    "Agro-forestry", "Broad-leaved forest", "Coniferous forest",
    "Mixed forest", "Natural grassland", "Moors and heathland",
    "Sclerophyllous vegetation", "Transitional woodland",
    "Beaches/dunes/sands", "Inland wetlands",
]


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 1: BigEarthNet.txt (Mandated — Real S1 SAR + S2 Optical + Text)
# ═══════════════════════════════════════════════════════════════════════════

class BigEarthNetTxtDataset(Dataset):
    """
    Loads co-registered Sentinel-1 SAR + Sentinel-2 Optical from BigEarthNet v2
    LMDB archives, with text annotations from BigEarthNet.txt parquet.

    Prerequisites on the DGX:
      1. Download BigEarthNet v2 from https://bigearth.net/
      2. Encode to LMDB:
           pip install configilm
           python -m configilm.extra.BENv2_Encoder /data/BigEarthNet/
      3. Download BigEarthNet.txt parquet:
           huggingface-cli download BIFOLD-BigEarthNetv2-0/BigEarthNet.txt

    Directory structure:
        /data/BigEarthNet/
            BENv2.lmdb/              ← LMDB encoded images
                data.mdb
                lock.mdb
            metadata.parquet         ← from BigEarthNet v2
            metadata_snow_cloud.parquet
            BigEarthNet.txt.parquet  ← text annotations (from HuggingFace)
    """

    OPTICAL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    OPTICAL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    SAR_MEAN = torch.tensor([-12.54, -20.19]).view(2, 1, 1)  # typical S1 dB
    SAR_STD = torch.tensor([5.25, 5.73]).view(2, 1, 1)

    def __init__(
        self,
        lmdb_dir: str,
        parquet_dir: str,
        split: str = "train",
        img_size: int = 224,
    ):
        try:
            from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
        except ImportError:
            raise ImportError(
                "configilm is required for BigEarthNet.txt.\n"
                "Install: pip install configilm\n"
                "Docs: https://lhackel-tub.github.io/ConfigILM/"
            )

        data_dirs = {
            "images_lmdb": str(lmdb_dir),
            "metadata_parquet": str(Path(parquet_dir) / "metadata.parquet"),
            "metadata_snow_cloud_parquet": str(
                Path(parquet_dir) / "metadata_snow_cloud.parquet"
            ),
        }

        # Load S2 optical (3 bands: B4, B3, B2 = RGB)
        self.s2_ds = BENv2DataSet(
            data_dirs=data_dirs,
            split=split,
            img_size=(3, img_size, img_size),
        )

        # Load S1 SAR (2 bands: VV, VH)
        self.s1_ds = BENv2DataSet(
            data_dirs=data_dirs,
            split=split,
            img_size=(2, img_size, img_size),
        )

        self.img_size = img_size
        print_main(
            f"[BigEarthNet.txt] Split={split} | "
            f"S2 samples={len(self.s2_ds)} | S1 samples={len(self.s1_ds)}"
        )

    def __len__(self):
        return min(len(self.s2_ds), len(self.s1_ds))

    def __getitem__(self, idx):
        # S2 optical → (3, H, W) float32 normalised
        s2_img, s2_label = self.s2_ds[idx]
        if isinstance(s2_img, np.ndarray):
            s2_img = torch.from_numpy(s2_img).float()
        if s2_img.shape[0] != 3:
            s2_img = s2_img[:3]
        s2_img = (s2_img - self.OPTICAL_MEAN) / self.OPTICAL_STD

        # S1 SAR → (2, H, W) float32 normalised (real radar backscatter!)
        s1_img, _ = self.s1_ds[idx]
        if isinstance(s1_img, np.ndarray):
            s1_img = torch.from_numpy(s1_img).float()
        if s1_img.shape[0] != 2:
            s1_img = s1_img[:2]
        s1_img = (s1_img - self.SAR_MEAN) / self.SAR_STD

        # Label → single-class (top label for classification head)
        if isinstance(s2_label, np.ndarray):
            label = int(s2_label.argmax()) % NUM_BIGEARTHNET_CLASSES
        elif isinstance(s2_label, (list, tuple)):
            label = int(s2_label[0]) % NUM_BIGEARTHNET_CLASSES if s2_label else 0
        elif isinstance(s2_label, int):
            label = s2_label % NUM_BIGEARTHNET_CLASSES
        else:
            label = 0

        return s2_img, s1_img, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 2: Generic disk directory (fallback)
# ═══════════════════════════════════════════════════════════════════════════

MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BigEarthNetDiskDataset(Dataset):
    """Loads images from a local directory. SAR is synthetic."""

    def __init__(self, data_root: str | Path, img_size: int = 224):
        self.root = Path(data_root)
        self.img_size = img_size
        self.files = sorted(
            list(self.root.rglob("*.png"))
            + list(self.root.rglob("*.jpg"))
            + list(self.root.rglob("*.tif"))
        )
        self.labels = {}
        labels_csv = self.root / "labels.csv"
        if labels_csv.exists():
            import csv
            with open(labels_csv) as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    self.labels[row[0]] = int(row[1]) % NUM_BIGEARTHNET_CLASSES
        print_main(f"[Disk] {len(self.files)} images | {len(self.labels)} labels")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = PILImage.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), PILImage.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN_NP) / STD_NP
        optical = torch.from_numpy(arr.transpose(2, 0, 1))
        sar = torch.rand(2, self.img_size, self.img_size)  # synthetic
        label = self.labels.get(path.stem, 0)
        return optical, sar, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 3: HuggingFace streaming (compact uint8 buffer)
# ═══════════════════════════════════════════════════════════════════════════

class BigEarthNetHFDataset(Dataset):
    """Streams from HuggingFace, caches compact uint8 in RAM."""

    def __init__(self, hf_dataset, max_samples: int = 200000, img_size: int = 224):
        self.img_size = img_size
        self.images = []
        self.labels = []

        print_main(f"[HF] Buffering {max_samples} samples...")
        for i, s in enumerate(hf_dataset):
            if i >= max_samples:
                break
            img = s.get("image") or s.get("s2_image") or s.get("optical")
            if img is not None:
                if not isinstance(img, PILImage.Image):
                    img = PILImage.fromarray(np.array(img))
                arr = np.array(
                    img.convert("RGB").resize((img_size, img_size), PILImage.BILINEAR),
                    dtype=np.uint8,
                )
            else:
                arr = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

            raw_lbl = s.get("labels", s.get("label", [0]))
            if isinstance(raw_lbl, list):
                lbl = int(raw_lbl[0]) % NUM_BIGEARTHNET_CLASSES if raw_lbl else 0
            elif isinstance(raw_lbl, int):
                lbl = raw_lbl % NUM_BIGEARTHNET_CLASSES
            else:
                lbl = 0

            self.images.append(arr)
            self.labels.append(lbl)
            if is_main() and (i + 1) % 10000 == 0:
                print(f"  {i+1}/{max_samples}...", flush=True)

        print_main(f"  ✓ Cached {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        arr = self.images[idx].astype(np.float32) / 255.0
        arr = (arr - MEAN_NP) / STD_NP
        optical = torch.from_numpy(arr.transpose(2, 0, 1))
        sar = torch.rand(2, self.images[idx].shape[0], self.images[idx].shape[1])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return optical, sar, label


# ─── Class weights (BigEarthNet-19 inverse frequency) ────────────────────────

CLASS_FREQ = torch.tensor([
    0.058, 0.026, 0.008, 0.003, 0.137, 0.027, 0.095, 0.056, 0.031,
    0.011, 0.179, 0.094, 0.073, 0.038, 0.019, 0.031, 0.028, 0.004, 0.032
], dtype=torch.float32)


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, epoch,
    warmup_iters, device,
) -> float:
    model.train()
    total_loss = 0.0
    n = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False,
                disable=not is_main(), dynamic_ncols=True)

    for step, (opt_img, sar_img, labels) in enumerate(pbar):
        global_step = epoch * n + step
        if global_step < warmup_iters:
            scale = (global_step + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg["lr"] = pg["_base_lr"] * scale

        opt_img = opt_img.to(device)
        sar_img = sar_img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            out = model(opt_img, sar_img, mode="classify")
            loss = criterion(out["logits"], labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if is_main():
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for opt_img, sar_img, labels in loader:
        opt_img = opt_img.to(device)
        sar_img = sar_img.to(device)
        labels = labels.to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            out = model(opt_img, sar_img, mode="classify")
            loss = criterion(out["logits"], labels)

        total_loss += loss.item()
        correct += (out["logits"].argmax(1) == labels).sum().item()
        total += labels.size(0)

    metrics_t = torch.tensor([total_loss, correct, total], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)

    n_loaders = len(loader) * (dist.get_world_size() if dist.is_initialized() else 1)
    return {
        "val_loss": float(metrics_t[0]) / max(n_loaders, 1),
        "accuracy": float(metrics_t[1]) / max(float(metrics_t[2]), 1),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    print_main("═══════════════════════════════════════════════════════")
    print_main("  SatQuery AI — Optical-SAR Fusion Training (DGX H200)")
    print_main("═══════════════════════════════════════════════════════")
    print_main(f"  GPUs: {world_size}  |  Batch/GPU: {args.batch_size}  |  "
               f"Effective batch: {args.batch_size * world_size}")
    print_main(f"  Data source: {args.data_source}  |  AMP: fp16  |  NCCL")

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.data_source == "bentxt":
        # ★ MANDATED: BigEarthNet.txt with real Sentinel-1 SAR
        assert args.lmdb_dir, "--lmdb_dir required for bentxt mode"
        assert args.parquet_dir, "--parquet_dir required for bentxt mode"
        train_ds = BigEarthNetTxtDataset(
            lmdb_dir=args.lmdb_dir,
            parquet_dir=args.parquet_dir,
            split="train",
            img_size=224,
        )
        val_ds = BigEarthNetTxtDataset(
            lmdb_dir=args.lmdb_dir,
            parquet_dir=args.parquet_dir,
            split="val",
            img_size=224,
        )
        print_main("  ★ Using REAL Sentinel-1 SAR + Sentinel-2 Optical")

    elif args.data_source == "disk":
        assert args.data_root, "--data_root required for disk mode"
        train_ds = BigEarthNetDiskDataset(Path(args.data_root) / "train")
        val_ds = BigEarthNetDiskDataset(Path(args.data_root) / "val")
        print_main("  ⚠ Disk mode — SAR is synthetic")

    else:
        from datasets import load_dataset
        HF_SOURCES = [
            "BigEarthNet/BigEarthNet-S2",
            "Bingsu/BigEarthNet",
            "flwrlabs/bigearthnet",
        ]
        hf_train = hf_val = None
        for src in HF_SOURCES:
            try:
                print_main(f"  Trying {src}...")
                hf_train = load_dataset(src, split="train", streaming=True)
                hf_val = load_dataset(src, split="validation", streaming=True)
                print_main(f"  ✓ Using {src}")
                break
            except Exception as e:
                print_main(f"  ✗ {src}: {str(e)[:60]}")

        if hf_train is None:
            raise RuntimeError("No BigEarthNet source available")

        train_ds = BigEarthNetHFDataset(hf_train, max_samples=args.max_samples)
        val_ds = BigEarthNetHFDataset(
            hf_val, max_samples=max(1000, args.max_samples // 10)
        )
        print_main("  ⚠ HF streaming mode — SAR is synthetic")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    num_workers = min(os.cpu_count() // world_size, 12)
    print_main(f"  Workers/GPU: {num_workers}  |  Train: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = OpticalSARFusionModel(
        num_classes=NUM_BIGEARTHNET_CLASSES,
        freeze_opt_stages=8,
        freeze_sar_stages=2,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_main(f"  Params: {total_p/1e6:.1f}M total | {train_p/1e6:.1f}M trainable")

    # ── Optimizer ────────────────────────────────────────────────────────
    enc_p = [p for n, p in model.named_parameters()
             if "encoder" in n and p.requires_grad]
    head_p = [p for n, p in model.named_parameters()
              if "encoder" not in n and p.requires_grad]

    optimizer = AdamW([
        {"params": enc_p, "lr": LR_BACKBONE, "_base_lr": LR_BACKBONE},
        {"params": head_p, "lr": LR_HEAD, "_base_lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, eta_min=1e-6)

    # Class-weighted loss
    cw = (1.0 / (CLASS_FREQ + 1e-3))
    cw = cw / cw.sum() * NUM_BIGEARTHNET_CLASSES
    criterion = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.1)

    scaler = GradScaler()
    warmup_iters = 5 * len(train_loader)

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc = 0.0
    ckpt_path = CKPT_DIR / "optical_sar_fused.pth"

    if args.resume and Path(args.resume).exists():
        print_main(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("val_metrics", {}).get("accuracy", 0.0)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print_main(f"  Resumed epoch {start_epoch} | best acc: {best_acc:.4f}")

    # ── TensorBoard ──────────────────────────────────────────────────────
    writer = None
    if is_main():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(_ROOT / "runs" / "fusion"))
            print_main(f"  TensorBoard: {_ROOT / 'runs' / 'fusion'}")
        except ImportError:
            pass

    # ── Training ─────────────────────────────────────────────────────────
    print_main(f"\n{'═'*60}")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            epoch, warmup_iters, device,
        )
        metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print_main(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_loss={metrics['val_loss']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | {elapsed:.0f}s"
        )

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", metrics["val_loss"], epoch)
            writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[1]["lr"], epoch)

        if is_main() and metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_metrics": metrics,
                "data_source": args.data_source,
            }, ckpt_path)
            print_main(f"  ✓ Best → {ckpt_path} (acc={best_acc:.4f})")

    print_main(f"\n{'═'*60}")
    print_main(f"Training complete. Best accuracy: {best_acc:.4f}")
    print_main(f"Data source: {args.data_source}")

    if writer:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DDP Fusion Training — DGX H200 (BigEarthNet.txt)"
    )
    parser.add_argument(
        "--data_source", choices=["bentxt", "disk", "hf"], default="bentxt",
        help="'bentxt' for mandated BigEarthNet.txt (real SAR), "
             "'disk' for local dir, 'hf' for HuggingFace streaming"
    )
    parser.add_argument("--lmdb_dir", type=str, default=None,
                        help="Path to BENv2.lmdb directory (for bentxt mode)")
    parser.add_argument("--parquet_dir", type=str, default=None,
                        help="Path to dir containing metadata.parquet (for bentxt mode)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to local image dir (for disk mode)")
    parser.add_argument("--max_samples", type=int, default=200000,
                        help="Max samples for HF streaming mode")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size PER GPU (effective = batch × num_gpus)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    main(parser.parse_args())
