from glob import glob
import os
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback

# COPY your current MineRLVideoDataset here, or we just define a new one later.
from train_tokenizer import MineRLVideoDataset  # if it's defined there

def main():
    roots = [
        "/Volumes/1TBSSD/MineRL/MineRLNavigate-v0",
        "/Volumes/1TBSSD/MineRL/MineRLTreechop-v1",
        "/Volumes/1TBSSD/MineRL/MineRLNavigateExtreme-v0",
        "/Volumes/1TBSSD/MineRL/MineRLObtainDiamond-v0",
    ]
    dataset = MineRLVideoDataset(roots, num_frames=32)
    print("len(dataset):", len(dataset))

    for idx in range(len(dataset)):
        print(f"Testing idx {idx} ...", end="", flush=True)
        try:
            v = dataset[idx]
            print(" OK, shape:", v[0,0,0,0:5])
        except Exception as e:
            print(" ERROR:", e)
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()
