import os
for filename in ["satquery/training/train_fusion.py", "satquery/training/train_fusion_ddp.py"]:
    if not os.path.exists(filename): continue
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace load_dataset(src, ...) with load_dataset(src, token=os.environ.get("HF_TOKEN"), ...)
    content = content.replace('load_dataset(src, split="train", streaming=True)', 
                              'load_dataset(src, split="train", streaming=True, token=os.environ.get("HF_TOKEN"))')
    content = content.replace('load_dataset(src, split="validation", streaming=True)', 
                              'load_dataset(src, split="validation", streaming=True, token=os.environ.get("HF_TOKEN"))')
    
    with open(filename, 'w') as f:
        f.write(content)
