"""Inspect a .pt checkpoint file to see its structure and keys."""
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "/data/lindseylm/GLM_EVALUATIONS/MODELS/MEGADNA/megaDNA/output/lambda_filtered/2k/megaDNA_lambda_filtered_2k_seed10_20260119_190848/classifier.pt"

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"Type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys: {list(v.keys())[:10]}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
else:
    print(f"Value: {ckpt}")
