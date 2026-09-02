from datasets import load_dataset
ds = load_dataset("GFM-Bench/BigEarthNet", split="train", streaming=True, trust_remote_code=True)
item = next(iter(ds))
print(item.keys())
