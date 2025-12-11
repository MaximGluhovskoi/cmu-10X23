import torch
from pathlib import Path

path = Path("/Volumes/1TBSSD/Downloads/data/f6daba428a5e19a3d47574858c13550499be23603422e6a0ee9728f8b53e192d/00e24ef06b5b4f3b/encoded_video/batch_0000.pt")  # change to whatever file you want

obj = torch.load(path, map_location="cpu")
print("Top-level type:", type(obj))

if isinstance(obj, dict):
    print("Keys:", obj.keys())
    for k, v in obj.items():
        print(f"\nKey: {k}")
        print("  type:", type(v))
        if isinstance(v, torch.Tensor):
            print("  shape:", tuple(v.shape), "dtype:", v.dtype)
        elif isinstance(v, (list, tuple)):
            print("  len:", len(v))
            if len(v) > 0:
                print("  first element type:", type(v[0]))
        else:
            print("  value preview:", repr(v)[:200])
elif isinstance(obj, torch.Tensor):
    print("Tensor shape:", tuple(obj.shape), "dtype:", obj.dtype)
else:
    print("Object repr:", repr(obj)[:500])
