"""
satquery/training/train_change_ddp.py
──────────────────────────────────────
Multi-GPU DDP training for the Siamese Change Detection U-Net on LEVIR-CD.
Optimised for NVIDIA DGX H200 (8× H200, 141 GB HBM3e each).

Features:
  • PyTorch DistributedDataParallel (NCCL backend)
  • Mixed-precision (torch.amp) — ~2× throughput on H200 Tensor Cores
  • Checkpoint resume (--resume flag)
  • TensorBoard logging
  • Proper num_workers scaling (auto = ncpus // ngpus)

Launch (8 GPUs on single DGX node):
    torchrun --nproc_per_node=8 satquery/training/train_change_ddp.py \\
        --data_root /data/LEVIR-CD \\
        --epochs 100 --batch_size 32 --img_size 512

Launch (4 GPUs):
    torchrun --nproc_per_node=4 satquery/training/train_change_ddp.py \\
        --data_root /data/LEVIR-CD \\
        --epochs 100 --batch_size 32

Resume from checkpoint:
    torchrun --nproc_per_node=8 satquery/training/train_change_ddp.py \\
        --data_root /data/LEVIR-CD --resume checkpoints/siamese_change.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from satquery.models.siamese_unet import SiameseUNet
from satquery.losses.compound_loss import CompoundChangeLoss
from satquery.training.train_change import LEVIRCDDataset, compute_iou, compute_f1
from satquery.config import (
    CKPT_DIR, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, CHANGE_THRESHOLD,
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


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CompoundChangeLoss,
    scaler: GradScaler,
    epoch: int,
    warmup_iters: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False,
                disable=not is_main(), dynamic_ncols=True)

    for step, (t1, t2, mask) in enumerate(pbar):
        # Linear warmup
        global_step = epoch * n + step
        if global_step < warmup_iters:
            lr_scale = (global_step + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg["lr"] = pg["_base_lr"] * lr_scale

        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with autocast(device_type="cuda", dtype=torch.float16):
            prob_map = model(t1, t2)
            logits = torch.logit(prob_map.clamp(1e-6, 1 - 1e-6))
            loss, components = criterion(logits, mask)

        # Scaled backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if is_main():
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "focal": f"{components['focal']:.4f}",
                "dice": f"{components['dice']:.4f}",
            })

    return total_loss / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ious, f1s = [], []
    for t1, t2, mask in loader:
        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
        with autocast(device_type="cuda", dtype=torch.float16):
            prob_map = model(t1, t2)
        ious.append(compute_iou(prob_map.cpu().float(), mask.cpu().float()))
        f1s.append(compute_f1(prob_map.cpu().float(), mask.cpu().float()))

    # Aggregate across GPUs
    iou_t = torch.tensor(np.mean(ious), device=device)
    f1_t = torch.tensor(np.mean(f1s), device=device)
    if dist.is_initialized():
        dist.all_reduce(iou_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(f1_t, op=dist.ReduceOp.AVG)
    return {"IoU": float(iou_t), "F1": float(f1_t)}


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    print_main(f"═══ DGX H200 Change Detection Training ═══")
    print_main(f"  GPUs: {world_size}  |  Batch/GPU: {args.batch_size}  |  "
               f"Effective batch: {args.batch_size * world_size}")
    print_main(f"  Epochs: {args.epochs}  |  Img size: {args.img_size}")
    print_main(f"  AMP: fp16  |  Backend: NCCL")

    # ── Dataset ──────────────────────────────────────────────────────────
    levir_root = Path(args.data_root)
    train_ds = LEVIRCDDataset(levir_root / "train", img_size=args.img_size, augment=True)
    val_ds = LEVIRCDDataset(levir_root / "val", img_size=args.img_size, augment=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=dist.get_rank(), shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=dist.get_rank(), shuffle=False)

    # Auto num_workers: total CPU cores / GPUs
    num_workers = min(os.cpu_count() // world_size, 12)
    print_main(f"  DataLoader workers/GPU: {num_workers}")

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
    model = SiameseUNet(in_channels=3, pretrained=True, freeze_stages=2).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_main(f"  Params: {total_p/1e6:.1f}M total | {train_p/1e6:.1f}M trainable")

    # ── Optimizer ────────────────────────────────────────────────────────
    encoder_params = [p for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if "encoder" not in n and p.requires_grad]

    optimizer = AdamW([
        {"params": encoder_params, "lr": LR_BACKBONE, "_base_lr": LR_BACKBONE},
        {"params": decoder_params, "lr": LR_HEAD, "_base_lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CompoundChangeLoss(focal_weight=0.5, dice_weight=0.5)
    scaler = GradScaler()
    warmup_iters = 3 * len(train_loader)

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_iou = 0.0
    ckpt_path = CKPT_DIR / "siamese_change.pth"

    if args.resume and Path(args.resume).exists():
        print_main(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_iou = ckpt.get("val_metrics", {}).get("IoU", 0.0)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print_main(f"  Resumed at epoch {start_epoch} | best IoU: {best_iou:.4f}")

    # ── TensorBoard ──────────────────────────────────────────────────────
    writer = None
    if is_main():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(_ROOT / "runs" / "change_detection"))
            print_main(f"  TensorBoard: {_ROOT / 'runs' / 'change_detection'}")
        except ImportError:
            print_main("  TensorBoard not available, skipping.")

    # ── Training loop ────────────────────────────────────────────────────
    print_main(f"\n{'═'*60}")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_sampler.set_epoch(epoch)  # reshuffle per epoch

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            epoch, warmup_iters, device,
        )
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print_main(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | IoU={metrics['IoU']:.4f} | "
            f"F1={metrics['F1']:.4f} | {elapsed:.0f}s"
        )

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/IoU", metrics["IoU"], epoch)
            writer.add_scalar("val/F1", metrics["F1"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[1]["lr"], epoch)

        # Save best (rank 0 only)
        if is_main() and metrics["IoU"] > best_iou:
            best_iou = metrics["IoU"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_metrics": metrics,
            }, ckpt_path)
            print_main(f"  ✓ Best checkpoint → {ckpt_path} (IoU={best_iou:.4f})")

    print_main(f"\n{'═'*60}")
    print_main(f"Training complete. Best IoU: {best_iou:.4f}")

    if writer:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Change Detection — DGX H200")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to LEVIR-CD root (must have train/val/test subdirs)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size PER GPU (effective = batch_size × num_gpus)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="512 recommended on H200 (141 GB VRAM)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    main(parser.parse_args())
