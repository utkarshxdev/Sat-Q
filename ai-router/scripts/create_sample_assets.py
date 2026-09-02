"""
scripts/create_sample_assets.py
───────────────────────────────
Generates realistic sample satellite image pairs in data/sample_images/
for testing and demonstrating the SatQuery AI web UI.
"""
from pathlib import Path
import numpy as np
from PIL import Image

sample_dir = Path("data/sample_images")
sample_dir.mkdir(parents=True, exist_ok=True)

h, w = 512, 512

# ── 1. Bi-Temporal Pair: Urban Development (T1 & T2) ─────────────────────────
t1 = np.zeros((h, w, 3), dtype=np.uint8)
t1[:, :, 1] = 160  # vegetation green
t1[:, :, 0] = 70
t1[:, :, 2] = 50

t2 = t1.copy()
# Add new concrete/residential blocks in T2 (grey/white)
t2[150:380, 150:400, :] = [210, 205, 200]
# Add road network
t2[:, 250:270, :] = [60, 60, 60]
t2[260:280, :, :] = [60, 60, 60]

Image.fromarray(t1).save(sample_dir / "bitemporal_T1_pre_urban.png")
Image.fromarray(t2).save(sample_dir / "bitemporal_T2_post_urban.png")

# ── 2. Bi-Temporal Pair: Flood Inundation (T1 & T2) ──────────────────────────
t1_flood = np.zeros((h, w, 3), dtype=np.uint8)
t1_flood[:, :, 1] = 180
t1_flood[:, :, 0] = 120
t1_flood[:, :, 2] = 60

t2_flood = t1_flood.copy()
# Inundated riverbed & flood plains (deep muddy blue)
t2_flood[100:350, :, :] = [30, 80, 150]

Image.fromarray(t1_flood).save(sample_dir / "bitemporal_T1_pre_flood.png")
Image.fromarray(t2_flood).save(sample_dir / "bitemporal_T2_post_flood.png")

# ── 3. Cross-Modal Pair: Optical + SAR Coastal Port ──────────────────────────
opt = np.zeros((h, w, 3), dtype=np.uint8)
opt[:, :256, :] = [20, 90, 180]    # Water on left
opt[:, 256:, :] = [180, 170, 150]  # Land/Urban on right
# Cloud cover over ocean (optical obscured)
opt[100:220, 50:200, :] = [240, 240, 245]

sar = np.zeros((h, w, 3), dtype=np.uint8)
sar[:, :256, :] = [15, 15, 20]     # Low backscatter on calm water
sar[:, 256:, :] = [190, 190, 200]  # High double-bounce on buildings
# Ship / metal structure visible through clouds on SAR
sar[140:180, 100:150, :] = [255, 255, 255]

Image.fromarray(opt).save(sample_dir / "optical_coastal_with_cloud.png")
Image.fromarray(sar).save(sample_dir / "sar_coastal_radar.png")

print(f"✓ Created 6 sample satellite images in {sample_dir.resolve()}:")
for f in sample_dir.iterdir():
    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
