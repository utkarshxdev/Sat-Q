"""
satquery/training/train_fusion.py
───────────────────────────────────
Fine-tuning script for the Optical-SAR Fusion Model on BigEarthNet.

Strategy (Colab-optimised):
  • Stream BigEarthNet directly from HuggingFace datasets (no 65 GB disk I/O).
  • Freeze the first 8 ViT blocks and first 2 ConvNeXt stages.
  • Train ONLY the cross-attention bottleneck + classifier head.
  • Expected Colab T4 training time: ~90 minutes for 5 epochs.

Usage (Colab):
    !pip install -q datasets transformers timm einops segmentation-models-pytorch
    !python satquery/training/train_fusion.py --epochs 5 --batch_size 16

Usage (local Mac M4 for sanity check with tiny subset):
    python satquery/training/train_fusion.py --epochs 1 --batch_size 4 --max_samples 128
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from satquery.config import (
    CKPT_DIR, DEVICE, BATCH_SIZE, LR_BACKBONE, LR_HEAD,
    WEIGHT_DECAY, MAX_EPOCHS, WARMUP_EPOCHS, NUM_BIGEARTHNET_CLASSES, IMG_SIZE,
)
from satquery.models.optical_sar_fusion import OpticalSARFusionModel


# ─── BigEarthNet Streaming Dataset ───────────────────────────────────────────

class BigEarthNetStreamDataset(Dataset):
    """
    Wraps HuggingFace streaming BigEarthNet dataset into a PyTorch Dataset.

    Columns expected (BigEarthNet-S1/S2 on HF):
        - 'image'        : PIL Image (Sentinel-2 RGB)
        - 'labels'       : list of int  (multi-hot, 19-class BigEarthNet-19)
        - 'sar_image'    : PIL Image (Sentinel-1 VV/VH) — if available

    Falls back to a random synthetic SAR if sar_image is unavailable
    (useful for testing with optical-only subsets).

    Args:
        hf_dataset   : HuggingFace IterableDataset (streaming).
        max_samples  : Limit number of samples (for local debugging).
        img_size     : Spatial resize target.
    """

    def __init__(self, hf_dataset, max_samples: Optional[int] = None, img_size: int = 224):
        self.img_size = img_size
        self.samples = []
        print("Buffering streaming dataset samples...")
        for i, sample in enumerate(hf_dataset):
            if max_samples and i >= max_samples:
                break
            self.samples.append(sample)
        print(f"  → Buffered {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # ── Optical image ──────────────────────────────────────────────────
        img = sample.get("image") or sample.get("s2_image")
        if img is None:
            optical = torch.rand(3, self.img_size, self.img_size)
        else:
            import torchvision.transforms.functional as TF
            from PIL import Image as PILImage
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(np.array(img))
            img = img.convert("RGB").resize((self.img_size, self.img_size))
            optical = TF.to_tensor(img)          # [0,1] float32, (3, H, W)
            # ImageNet normalise
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            optical = (optical - mean) / std

        # ── SAR image (synthetic fallback) ─────────────────────────────────
        sar_raw = sample.get("sar_image") or sample.get("s1_image")
        if sar_raw is None:
            sar = torch.rand(2, self.img_size, self.img_size)
        else:
            from PIL import Image as PILImage
            if not isinstance(sar_raw, PILImage.Image):
                sar_raw = PILImage.fromarray(np.array(sar_raw))
            sar_arr = np.array(
                sar_raw.resize((self.img_size, self.img_size))
            ).astype(np.float32) / 255.0
            if sar_arr.ndim == 2:
                sar_arr = np.stack([sar_arr, sar_arr], axis=0)
            else:
                sar_arr = sar_arr.transpose(2, 0, 1)[:2]
            sar = torch.from_numpy(sar_arr)

        # ── Labels (multi-hot) ─────────────────────────────────────────────
        raw_labels = sample.get("labels", [])
        label_vec = torch.zeros(NUM_BIGEARTHNET_CLASSES)
        for lbl in raw_labels:
            if isinstance(lbl, int) and 0 <= lbl < NUM_BIGEARTHNET_CLASSES:
                label_vec[lbl] = 1.0

        return optical, sar, label_vec


from tqdm import tqdm

def train_one_epoch(
    model: OpticalSARFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    warmup_iters: int,
    base_lr_head: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, dynamic_ncols=True)
    for step, (optical, sar, labels) in enumerate(pbar):
        # ── Linear warmup ──────────────────────────────────────────────────
        global_step = epoch * n_batches + step
        if global_step < warmup_iters:
            lr_scale = (global_step + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("_base_lr", base_lr_head) * lr_scale

        optical = optical.to(DEVICE)
        sar     = sar.to(DEVICE)
        labels  = labels.to(DEVICE)

        optimizer.zero_grad()
        output  = model(optical, sar, mode="classify")
        loss    = criterion(output["logits"], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/(step+1):.4f}"})

    return total_loss / n_batches


def evaluate(
    model: OpticalSARFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for optical, sar, labels in loader:
            optical, sar, labels = (
                optical.to(DEVICE), sar.to(DEVICE), labels.to(DEVICE)
            )
            output = model(optical, sar, mode="classify")
            loss   = criterion(output["logits"], labels)
            total_loss += loss.item()

            preds = torch.sigmoid(output["logits"]).cpu().numpy() > 0.5
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    import numpy as np
    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)

    # Micro-F1
    tp = (preds_arr & labels_arr.astype(bool)).sum()
    fp = (preds_arr & ~labels_arr.astype(bool)).sum()
    fn = (~preds_arr & labels_arr.astype(bool)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "val_loss": total_loss / len(loader),
        "micro_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main(args: argparse.Namespace) -> None:
    print(f"[Device] {DEVICE}")
    print(f"[Config] epochs={args.epochs}, batch={args.batch_size}, "
          f"max_samples={args.max_samples}")

    # ── Load dataset ──────────────────────────────────────────────────────
    hf_train = hf_val = None
    dataset_candidates = [
        "Bingsu/BigEarthNet",
        "flwrlabs/bigearthnet",
        "BigEarthNet/BigEarthNet-S2",
    ]
    try:
        from datasets import load_dataset
        for ds_name in dataset_candidates:
            try:
                print(f"Trying to stream {ds_name} from HuggingFace...")
                hf_train = load_dataset(ds_name, split="train", streaming=True, trust_remote_code=True)
                hf_val   = load_dataset(ds_name, split="validation", streaming=True, trust_remote_code=True)
                print(f"  ✓ Successfully connected to {ds_name}")
                break
            except Exception as candidate_err:
                print(f"  [Info] {ds_name} unavailable: {candidate_err}")
                hf_train = hf_val = None
    except Exception as e:
        print(f"[Warning] HuggingFace datasets library issue: {e}")

    if hf_train is not None:
        train_ds = BigEarthNetStreamDataset(
            hf_train, max_samples=args.max_samples, img_size=224
        )
        val_ds   = BigEarthNetStreamDataset(
            hf_val, max_samples=max(64, args.max_samples // 8), img_size=224
        )
    else:
        # Synthetic fallback — verifies shapes and training loop
        class SyntheticDS(Dataset):
            def __len__(self): return 256
            def __getitem__(self, _):
                return (
                    torch.randn(3, 224, 224),
                    torch.rand(2, 224, 224),
                    (torch.rand(NUM_BIGEARTHNET_CLASSES) > 0.7).float(),
                )
        train_ds = val_ds = SyntheticDS()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = OpticalSARFusionModel(
        opt_pretrained=True,
        sar_pretrained=True,
        freeze_opt_stages=12,
        freeze_sar_stages=4,
    ).to(DEVICE)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M", flush=True)

    # ── Optimizer: separate LRs for backbone vs. head ────────────────────
    backbone_params = [
        p for n, p in model.named_parameters()
        if ("encoder" in n) and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if ("encoder" not in n) and p.requires_grad
    ]
    optimizer = AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "_base_lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD,     "_base_lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    warmup_iters = WARMUP_EPOCHS * len(train_loader)

    best_f1 = 0.0
    ckpt_path = CKPT_DIR / "optical_sar_fused.pth"

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            epoch, warmup_iters, LR_HEAD,
        )
        metrics = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"micro_f1={metrics['micro_f1']:.4f} | "
            f"time={elapsed:.0f}s",
            flush=True
        )

        if metrics["micro_f1"] >= best_f1:
            best_f1 = metrics["micro_f1"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": metrics,
                },
                ckpt_path,
            )
            print(f"  ✓ Best checkpoint saved → {ckpt_path}", flush=True)

    print(f"\nTraining complete. Best micro-F1: {best_f1:.4f}", flush=True)


# ─── Public aliases for notebook imports ─────────────────────────────────────

class SyntheticFusionDataset(Dataset):
    """Synthetic fallback dataset — shape verification only. Single-label mode."""
    def __init__(self, n: int = 256, img_size: int = 224):
        self.n = n
        self.s = img_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _):
        label = torch.randint(0, NUM_BIGEARTHNET_CLASSES, (1,)).item()
        return (
            torch.randn(3, self.s, self.s),
            torch.rand(2, self.s, self.s),
            torch.tensor(label, dtype=torch.long),
        )


def evaluate_fusion(model, loader, device) -> dict:
    """Simple top-1 accuracy evaluation for single-label classification."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for optical, sar, labels in loader:
            optical, sar, labels = optical.to(device), sar.to(device), labels.to(device)
            out = model(optical, sar, mode="classify")
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return {"accuracy": correct / max(total, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Optical-SAR Fusion Model")
    parser.add_argument("--epochs",      type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max training samples (use small value locally).")
    main(parser.parse_args())

