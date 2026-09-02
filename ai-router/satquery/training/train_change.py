"""
satquery/training/train_change.py
───────────────────────────────────
Fine-tuning script for the Siamese Change Detection UNet on LEVIR-CD.

LEVIR-CD is a ~1 GB dataset of bi-temporal aerial/satellite images with
binary change masks. It is the standard benchmark for change detection and
can be downloaded locally on Mac without disk issues.

Dataset structure expected:
    data/LEVIR-CD/
        train/
            A/         ← pre-change images  (*.png)
            B/         ← post-change images (*.png)
            label/     ← binary masks       (*.png, 255=change, 0=no-change)
        val/
            A/ B/ label/
        test/
            A/ B/ label/

Download:
    https://justchenhao.github.io/LEVIR/   (~450 MB)
    Or: kaggle datasets download -d justchenhao/levir-cd

Usage (local Mac M4):
    python satquery/training/train_change.py --epochs 30 --batch_size 4

Usage (Colab T4):
    python satquery/training/train_change.py --epochs 50 --batch_size 16
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

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
from PIL import Image

from satquery.config import (
    CKPT_DIR, DEVICE, BATCH_SIZE, LR_BACKBONE, LR_HEAD,
    WEIGHT_DECAY, MAX_EPOCHS, WARMUP_EPOCHS, CHANGE_THRESHOLD,
)
from satquery.models.siamese_unet import SiameseUNet
from satquery.losses.compound_loss import CompoundChangeLoss


# ─── LEVIR-CD Dataset ─────────────────────────────────────────────────────────

class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD bi-temporal change detection dataset.

    Args:
        root      : Path to LEVIR-CD split dir (e.g. data/LEVIR-CD/train).
        img_size  : Crop/resize target (default 256 to fit Mac RAM).
        augment   : Apply random flips and rotations (training only).
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root: str | Path,
        img_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.root     = Path(root)
        self.img_size = img_size
        self.augment  = augment

        self.a_dir    = self.root / "A"
        self.b_dir    = self.root / "B"
        self.lbl_dir  = self.root / "label"

        if not self.a_dir.exists():
            raise FileNotFoundError(
                f"LEVIR-CD directory not found: {self.a_dir}\n"
                "Download from https://justchenhao.github.io/LEVIR/ and place under "
                f"{self.root.parent}"
            )

        self.file_names: List[str] = sorted(
            f.name for f in self.a_dir.iterdir() if f.suffix in (".png", ".jpg", ".tif")
        )
        print(f"[LEVIR-CD] {self.root.name} split: {len(self.file_names)} image pairs")

    def __len__(self) -> int:
        return len(self.file_names)

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load image → (3, H, W) float32 normalised."""
        img = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = (arr - self.MEAN) / self.STD               # ImageNet normalise
        return arr.transpose(2, 0, 1)                    # (3, H, W)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load binary mask → (1, H, W) float32 {0, 1}."""
        mask = Image.open(path).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        return arr[np.newaxis, ...]   # (1, H, W)

    def _augment(
        self, t1: np.ndarray, t2: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consistent random flips across T1, T2, and mask."""
        if random.random() > 0.5:
            t1, t2, mask = (
                np.flip(t1, axis=2).copy(),
                np.flip(t2, axis=2).copy(),
                np.flip(mask, axis=2).copy(),
            )
        if random.random() > 0.5:
            t1, t2, mask = (
                np.flip(t1, axis=1).copy(),
                np.flip(t2, axis=1).copy(),
                np.flip(mask, axis=1).copy(),
            )
        return t1, t2, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.file_names[idx]
        t1   = self._load_rgb(self.a_dir   / name)
        t2   = self._load_rgb(self.b_dir   / name)
        mask = self._load_mask(self.lbl_dir / name)

        if self.augment:
            t1, t2, mask = self._augment(t1, t2, mask)

        return (
            torch.from_numpy(t1).float(),
            torch.from_numpy(t2).float(),
            torch.from_numpy(mask).float(),
        )


class SyntheticChangeDataset(Dataset):
    """Synthetic fallback when LEVIR-CD is not present. For shape verification only."""

    def __init__(self, n: int = 128, img_size: int = 256):
        self.n = n
        self.s = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        t1   = torch.randn(3, self.s, self.s)
        t2   = torch.randn(3, self.s, self.s)
        mask = (torch.rand(1, self.s, self.s) > 0.95).float()  # ~5% change
        return t1, t2, mask


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Binary IoU (Intersection over Union) for change class."""
    pred = (pred_mask > CHANGE_THRESHOLD).bool()
    gt   = gt_mask.bool()
    intersection = (pred & gt).float().sum()
    union        = (pred | gt).float().sum()
    return float(intersection / (union + 1e-8))


def compute_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred = (pred_mask > CHANGE_THRESHOLD).bool()
    gt   = gt_mask.bool()
    tp = (pred & gt).float().sum()
    fp = (pred & ~gt).float().sum()
    fn = (~pred & gt).float().sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


# ─── Training ────────────────────────────────────────────────────────────────

from tqdm import tqdm

def train_one_epoch(
    model: SiameseUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CompoundChangeLoss,
    epoch: int,
    warmup_iters: int,
    base_lr: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, dynamic_ncols=True)
    for step, (t1, t2, mask) in enumerate(pbar):
        global_step = epoch * n + step
        if global_step < warmup_iters:
            lr_scale = (global_step + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("_base_lr", base_lr) * lr_scale

        t1, t2, mask = t1.to(DEVICE), t2.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        prob_map = model(t1, t2)              # (B, 1, H, W) ∈ [0,1]
        # CompoundChangeLoss expects logits — convert back via logit()
        logits = torch.logit(prob_map.clamp(1e-6, 1 - 1e-6))
        loss, components = criterion(logits, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"total": f"{components['total']:.4f}", "focal": f"{components['focal']:.4f}", "dice": f"{components['dice']:.4f}"})

    return total_loss / n


def evaluate(
    model: SiameseUNet,
    loader: DataLoader,
) -> dict:
    model.eval()
    ious, f1s = [], []

    with torch.no_grad():
        for t1, t2, mask in loader:
            t1, t2, mask = t1.to(DEVICE), t2.to(DEVICE), mask.to(DEVICE)
            prob_map = model(t1, t2)
            ious.append(compute_iou(prob_map.cpu(), mask.cpu()))
            f1s.append(compute_f1(prob_map.cpu(), mask.cpu()))

    return {
        "IoU": float(np.mean(ious)),
        "F1":  float(np.mean(f1s)),
    }


def main(args: argparse.Namespace) -> None:
    print(f"[Device] {DEVICE}")

    levir_root = Path(args.data_root)
    train_root = levir_root / "train"
    val_root   = levir_root / "val"

    try:
        train_ds = LEVIRCDDataset(train_root, img_size=args.img_size, augment=True)
        val_ds   = LEVIRCDDataset(val_root,   img_size=args.img_size, augment=False)
    except FileNotFoundError as e:
        print(f"[Warning] {e}")
        print("Using synthetic dataset for shape/training loop verification.")
        train_ds = SyntheticChangeDataset(256, args.img_size)
        val_ds   = SyntheticChangeDataset(64,  args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SiameseUNet(in_channels=3, pretrained=True, freeze_stages=2).to(DEVICE)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total: {total_p/1e6:.1f}M | Trainable: {trainable_p/1e6:.1f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────
    encoder_params = [p for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if "encoder" not in n and p.requires_grad]

    optimizer = AdamW([
        {"params": encoder_params, "lr": LR_BACKBONE, "_base_lr": LR_BACKBONE},
        {"params": decoder_params, "lr": LR_HEAD,     "_base_lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CompoundChangeLoss(focal_weight=0.5, dice_weight=0.5)
    warmup_iters = WARMUP_EPOCHS * len(train_loader)

    best_iou  = 0.0
    ckpt_path = CKPT_DIR / "siamese_change.pth"

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            epoch, warmup_iters, LR_HEAD,
        )
        metrics = evaluate(model, val_loader)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"IoU={metrics['IoU']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"time={elapsed:.0f}s"
        )

        if metrics["IoU"] > best_iou:
            best_iou = metrics["IoU"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": metrics,
                },
                ckpt_path,
            )
            print(f"  ✓ Best checkpoint saved → {ckpt_path}  (IoU={best_iou:.4f})")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Change Detection UNet")
    parser.add_argument("--data_root",  type=str, default="data/LEVIR-CD")
    parser.add_argument("--epochs",     type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img_size",   type=int, default=256,
                        help="Crop size (256 for Mac, 512 for Colab).")
    main(parser.parse_args())
