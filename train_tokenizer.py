# train_tokenizer.py
import argparse
import os
from glob import glob

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer_ms_vqvae import MultiScaleVQTokenizerVideo2D as Tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MineRLVideoDataset(Dataset):
    """
    Each item: random num_frames clip from recording.mp4 in a trajectory folder.

    Returns:
        video: (3, T, 64, 64) float32 in [-1,1]
    """

    def __init__(self, data_root: str, num_frames: int = 32):
        super().__init__()
        self.num_frames = num_frames
        self.clip_dirs = sorted(
            [p for p in glob(os.path.join(data_root, "*")) if os.path.isdir(p)]
        )

        self.index = []  # list of (dir_path, n_frames)
        for d in self.clip_dirs:
            mp4_path = os.path.join(d, "recording.mp4")
            if not os.path.exists(mp4_path):
                continue
            reader = imageio.get_reader(mp4_path)
            n_frames = reader.count_frames()
            reader.close()

            if n_frames < num_frames:
                continue

            self.index.append((d, n_frames))

        if not self.index:
            raise RuntimeError("No valid trajectories with enough frames found in data_root")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dir_path, n_frames = self.index[idx]
        mp4_path = os.path.join(dir_path, "recording.mp4")

        start = np.random.randint(0, n_frames - self.num_frames + 1)
        end = start + self.num_frames

        reader = imageio.get_reader(mp4_path)
        frames = []
        for t in range(start, end):
            frame = reader.get_data(t)  # (64,64,3) uint8
            frames.append(frame)
        reader.close()

        video = np.stack(frames, axis=0)  # (T,H,W,3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0  # (3,T,H,W)
        video = video * 2.0 - 1.0  # [-1,1]
        return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_ckpt", type=str, default="tokenizer_ms_vqvae.pt")
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    dataset = MineRLVideoDataset(args.data_root, num_frames=args.num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = Tokenizer(
        in_channels=3,
        base_channels=64,
        latent_dim=128,
        num_embeddings=256,
        commitment_cost=0.02,
    ).to(device)

    # Slightly higher LR to help the encoder learn; should still be stable on MPS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for vids in loader:
            vids = vids.to(device)  # (B,3,T,64,64)

            tokens, vq_loss, z_q = model.encode_tokens(vids)
            # decode from quantized latents so gradients reach encoder/VQ
            recon = model.decode_latents(z_q)
            # keep outputs in range for stable loss
            recon = torch.tanh(recon)
            recon_loss = F.mse_loss(recon, vids)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.save_ckpt)
            print(f"New best ({best:.4f}), saved tokenizer to {args.save_ckpt}")


if __name__ == "__main__":
    main()
