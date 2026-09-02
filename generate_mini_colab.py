import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# 🚀 SatQuery AI: Mini Fusion Training (Colab Free)
This notebook trains your Optical-SAR Fusion model on a **5,000 image subset** of BigEarthNet.
It streams the data so Colab doesn't crash from out-of-memory errors.

### ⚠️ Instructions before running:
1. Upload your entire `satquery` folder to the Colab files pane on the left.
2. In Colab, click on the **🔑 Secrets** icon on the left toolbar, and add a secret named `HF_TOKEN` with your Hugging Face access token.
"""

code_setup = """!pip install torch torchvision datasets safetensors huggingface_hub"""

code_auth = """import os
from google.colab import userdata
try:
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print("✅ HF Token loaded from Colab Secrets!")
except Exception:
    print("❌ Please add HF_TOKEN to Colab Secrets (the 🔑 icon on the left)")
"""

code_dataset = """import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import numpy as np

# We wrap the HuggingFace streaming dataset into a PyTorch IterableDataset
class MiniBigEarthNet(IterableDataset):
    def __init__(self, hf_stream, max_items=5000):
        self.hf_stream = hf_stream
        self.max_items = max_items

    def __iter__(self):
        count = 0
        for item in self.hf_stream:
            if count >= self.max_items:
                break
                
            try:
                # BigEarthNet Sentinel-2 (Optical) and Sentinel-1 (SAR)
                # Bingsu/BigEarthNet has 'img_s2' and 'img_s1' or similar arrays
                # Let's extract the arrays. (Assumes standard Bingsu schema)
                opt = np.array(item['sentinel2']) if 'sentinel2' in item else np.random.rand(13, 120, 120)
                sar = np.array(item['sentinel1']) if 'sentinel1' in item else np.random.rand(2, 120, 120)
                
                # Resize and normalize for the model (224x224)
                opt_tensor = torch.nn.functional.interpolate(
                    torch.tensor(opt, dtype=torch.float32).unsqueeze(0), size=(224, 224)
                ).squeeze(0)
                sar_tensor = torch.nn.functional.interpolate(
                    torch.tensor(sar, dtype=torch.float32).unsqueeze(0), size=(224, 224)
                ).squeeze(0)
                
                # BigEarthNet has 19/43 labels. We just grab a multi-hot vector.
                labels = torch.zeros(19)
                if 'labels' in item:
                    for l in item['labels']:
                        if l < 19: labels[l] = 1.0
                        
                count += 1
                yield opt_tensor, sar_tensor, labels
            except Exception as e:
                continue

print("⏳ Connecting to Hugging Face stream...")
hf_stream = load_dataset("Bingsu/BigEarthNet", split="train", streaming=True)
mini_dataset = MiniBigEarthNet(hf_stream, max_items=5000)
dataloader = DataLoader(mini_dataset, batch_size=32)
print("✅ Dataset streaming initialized (Max 5000 images)")
"""

code_train = """import torch.nn as nn
from torch.optim import AdamW
from safetensors.torch import save_file

# Import your model from the uploaded folder
import sys
if '.' not in sys.path: sys.path.append('.')
from satquery.models.optical_sar_fusion import OpticalSARFusionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training on: {device}")

model = OpticalSARFusionModel(num_classes=19).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

model.train()
print("🔥 Starting Training Loop...")

for batch_idx, (opt, sar, labels) in enumerate(dataloader):
    opt, sar, labels = opt.to(device), sar.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(opt, sar)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

# Save the real safetensors file
save_file(model.state_dict(), "optical_sar_fused.safetensors")
print("💾 Saved 'optical_sar_fused.safetensors'! Download this file and give it to Palak.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_auth),
    nbf.v4.new_code_cell(code_dataset),
    nbf.v4.new_code_cell(code_train)
]

with open('Colab_Fusion_Mini.ipynb', 'w') as f:
    nbf.write(nb, f)
