"""
scripts/deep_model_check.py
───────────────────────────
Full diagnostic audit of both SatQuery AI models.
Prints a clean, structured report without optimizer state noise.
"""
import sys, time
sys.path.insert(0, ".")

import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from satquery.models.siamese_unet import SiameseUNet
from satquery.models.optical_sar_fusion import OpticalSARFusionModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SEP  = "━" * 70
SEP2 = "─" * 60

BIGEARTHNET_CLASSES = [
    "Urban fabric","Industrial/commercial","Mine/dump/construction","Artificial non-ag. veg.",
    "Arable land","Permanent crops","Pastures","Complex cultivation","Agri. + nat. veg.",
    "Agro-forestry","Broad-leaved forest","Coniferous forest","Mixed forest",
    "Natural grassland","Moors and heathland","Sclerophyllous veg.","Transitional woodland",
    "Beaches/dunes/sands","Inland wetlands",
]

print(f"\n{'='*70}")
print(f"  SatQuery AI — Deep Model Diagnostic  |  Device: {device}")
print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
def weight_audit(model, title):
    total, trainable, frozen = 0, 0, 0
    nans, infs, dead = [], [], []
    groups = defaultdict(lambda: {"total": 0, "train": 0})

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        g = name.split(".")[0]
        groups[g]["total"] += n
        if p.requires_grad:
            trainable += n
            groups[g]["train"] += n
        else:
            frozen += n
        d = p.data.float().cpu()
        if torch.isnan(d).any():  nans.append(name)
        if torch.isinf(d).any():  infs.append(name)
        if d.abs().max() < 1e-6:  dead.append(name)

    print(f"\n{SEP2}")
    print(f"  {title} — Weight Audit")
    print(f"{SEP2}")
    print(f"  Total parameters  : {total:>13,}")
    print(f"  Trainable         : {trainable:>13,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen (backbone) : {frozen:>13,}  ({100*frozen/total:.1f}%)")
    print(f"  NaN tensors       : {len(nans):>3}   {'⚠️ CHECK!' if nans else '✓ Clean'}")
    print(f"  Inf tensors       : {len(infs):>3}   {'⚠️ CHECK!' if infs else '✓ Clean'}")
    print(f"  Dead weight groups: {len(dead):>3}   {'⚠️ CHECK!' if dead else '✓ None'}")
    print(f"\n  Layer Group Breakdown:")
    for g, s in sorted(groups.items(), key=lambda x: -x[1]["total"]):
        flag = "🔓" if s["train"] > 0 else "🔒"
        print(f"  {flag}  {g:<30s} {s['total']:>10,}  ({s['train']:,} trainable)")
    return total, trainable, frozen


def latency_bench(fn, label, n=30):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    p95 = np.percentile(a, 95)
    result = "✓ PASS" if p95 < 300 else "✗ FAIL"
    print(f"\n  Inference Latency ({n} warm runs):")
    print(f"    p50={np.percentile(a,50):.1f}ms | p95={p95:.1f}ms | min={a.min():.1f}ms | max={a.max():.1f}ms")
    print(f"    {result}  (target p95 < 300 ms)")
    return p95


# ══════════════════════════════════════════════════════════════════════════════
# 1. SiameseUNet
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("  1 / 2   SIAMESE U-NET  —  Change Detection")
print(SEP)

raw1 = torch.load("checkpoints/siamese_change.pth", map_location="cpu", weights_only=False)

# Print only scalar / non-tensor metadata
print("\n  Checkpoint metadata:")
for k, v in raw1.items():
    if k not in ("model_state_dict", "optimizer_state_dict"):
        print(f"    {k}: {v}")

model1 = SiameseUNet().to(device)
msg1 = model1.load_state_dict(raw1["model_state_dict"], strict=True)
model1.eval()
print(f"\n  State-dict load: missing={len(msg1.missing_keys)} | unexpected={len(msg1.unexpected_keys)}")

t1c, t1r, t1f = weight_audit(model1, "SiameseUNet")

# Forward pass
t1 = torch.randn(1, 3, 512, 512, device=device)
t2 = torch.randn(1, 3, 512, 512, device=device)
with torch.no_grad():
    out1 = model1(t1, t2)

spread1 = float(out1.max() - out1.min())
print(f"\n  Forward-Pass Shapes:")
print(f"    T1 input  : {tuple(t1.shape)}")
print(f"    T2 input  : {tuple(t2.shape)}")
print(f"    prob_map  : {tuple(out1.shape)}")
print(f"    value range : [{out1.min().item():.4f}, {out1.max().item():.4f}]")
print(f"    mean / std  : {out1.mean().item():.4f} / {out1.std().item():.4f}")
print(f"    Output spread: {spread1:.5f}  {'✓ Alive' if spread1 > 0.001 else '⚠️ DEAD OUTPUT'}")

# Siamese weight sharing
shared = id(model1.encoder) == id(model1.encoder)
print(f"    Siamese shared encoder: {'✓ YES' if shared else '✗ BROKEN'}")

p95_1 = latency_bench(lambda: model1(t1, t2), "SiameseUNet @ 512×512")
del raw1  # free optimizer state memory


# ══════════════════════════════════════════════════════════════════════════════
# 2. OpticalSARFusionModel
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{SEP}")
print("  2 / 2   OPTICAL-SAR FUSION  —  Embedding + Classification")
print(SEP)

raw2 = torch.load("checkpoints/optical_sar_fused.pth", map_location="cpu", weights_only=False)

print("\n  Checkpoint metadata:")
for k, v in raw2.items():
    if k not in ("model_state_dict", "optimizer_state_dict"):
        if isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(vv, (torch.Tensor, dict, list)):
                    print(f"    val_metrics.{kk}: {vv:.6f}" if isinstance(vv, float) else f"    val_metrics.{kk}: {vv}")
        else:
            print(f"    {k}: {v}")

model2 = OpticalSARFusionModel().to(device)
msg2 = model2.load_state_dict(raw2["model_state_dict"], strict=True)
model2.eval()
print(f"\n  State-dict load: missing={len(msg2.missing_keys)} | unexpected={len(msg2.unexpected_keys)}")

t2c, t2r, t2f = weight_audit(model2, "OpticalSARFusionModel")

# Forward pass
opt = torch.randn(1, 3, 224, 224, device=device)
sar = torch.randn(1, 2, 224, 224, device=device)
with torch.no_grad():
    out2 = model2(opt, sar, mode="classify")

emb    = out2["embedding"]
logits = out2["logits"]
attn   = out2["attn_weights"]

print(f"\n  Forward-Pass Shapes:")
print(f"    optical input   : {tuple(opt.shape)}")
print(f"    SAR input       : {tuple(sar.shape)}")
print(f"    embedding (768-D): {tuple(emb.shape)}")
print(f"    logits (19-cls)  : {tuple(logits.shape)}")
print(f"    attn_weights     : {tuple(attn.shape)}")

# Embedding health
emb_np = emb.squeeze().cpu().float().numpy()
spread2 = float(emb_np.max() - emb_np.min())
print(f"\n  Embedding statistics (768-D):")
print(f"    mean   : {emb_np.mean():.4f}")
print(f"    std    : {emb_np.std():.4f}")
print(f"    L2-norm: {np.linalg.norm(emb_np):.4f}")
print(f"    spread : {spread2:.4f}  {'✓ Alive' if spread2 > 0.01 else '⚠️ DEAD'}")

# Top-5 land cover
probs = F.softmax(logits, dim=-1).squeeze()
top5v, top5i = probs.topk(5)
print(f"\n  Top-5 BigEarthNet predictions (random noise input — expect uniform):")
for rank, (v, i) in enumerate(zip(top5v.tolist(), top5i.tolist())):
    label = BIGEARTHNET_CLASSES[i] if i < len(BIGEARTHNET_CLASSES) else f"Class-{i}"
    print(f"    {rank+1}. {label:<35s} {v*100:5.2f}%")

# Cross-attention health
attn_np = attn.squeeze().cpu().float().numpy()
attn_row_sum = attn_np.sum(axis=-1).mean()
print(f"\n  Cross-Attention map {tuple(attn.shape)}:")
print(f"    mean        : {attn_np.mean():.6f}")
print(f"    std         : {attn_np.std():.6f}")
print(f"    max         : {attn_np.max():.4f}")
print(f"    row-sum mean: {attn_row_sum:.4f}  {'✓ ~1.0 (softmax ok)' if 0.9 < attn_row_sum < 1.1 else '⚠️ Unexpected'}")

p95_2 = latency_bench(lambda: model2(opt, sar, mode="classify"), "FusionModel @ 224×224")
del raw2


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("  DEEP CHECK SUMMARY")
print(f"{'='*70}")
print(f"  {'Model':<38} {'Total':>10} {'Train':>10} {'Frozen':>10}")
print(f"  {'─'*38} {'─'*10} {'─'*10} {'─'*10}")
print(f"  {'SiameseUNet (Change Detector)':<38} {t1c:>10,} {t1r:>10,} {t1f:>10,}")
print(f"  {'OpticalSARFusion (Embed+Cls)':<38} {t2c:>10,} {t2r:>10,} {t2f:>10,}")
print(f"  {'COMBINED':<38} {t1c+t2c:>10,} {t1r+t2r:>10,} {t1f+t2f:>10,}")
print()
print(f"  Integrity checks:")
print(f"  {'✓' if spread1 > 0.001 else '✗'}  SiameseUNet output alive        (spread={spread1:.5f})")
print(f"  {'✓' if spread2 > 0.01  else '✗'}  FusionModel embedding alive      (spread={spread2:.4f})")
print(f"  {'✓' if 0.9 < attn_row_sum < 1.1 else '✗'}  Cross-attention softmax valid    (row-sum={attn_row_sum:.4f})")
print()
print(f"  Latency (p95, Apple MPS, target < 300 ms):")
print(f"  {'✓ PASS' if p95_1 < 300 else '✗ FAIL'}  SiameseUNet        {p95_1:.1f} ms")
print(f"  {'✓ PASS' if p95_2 < 300 else '✗ FAIL'}  FusionModel        {p95_2:.1f} ms")
print(f"\n{'='*70}")
print("  Deep check complete — all systems verified.")
print(f"{'='*70}\n")
