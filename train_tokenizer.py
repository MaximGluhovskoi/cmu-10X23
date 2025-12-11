# train_tokenizer.py
import argparse
import os
from glob import glob
import json

from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from tokenizer_ms_vqvae import MultiScaleVQTokenizerVideo2D as Tokenizer

import wandb

def describe_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Codebook size (VQ embeddings)
    vq_params = model.vq.embedding.num_embeddings * model.vq.embedding.embedding_dim

    print("\n===== TOKENIZER MODEL SUMMARY =====")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Embedding table:      {model.vq.embedding.num_embeddings} x {model.vq.embedding.embedding_dim}  "
          f"= {vq_params:,} parameters")
    print(f"Estimated memory:     {total_params * 4 / 1e6:.2f} MB  (FP32)")
    print("===================================\n")



def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MineRLVideoDataset(Dataset):
    """
    Each item: random num_frames clip from recording.mp4 in a trajectory folder.

    Uses metadata.json["true_video_frame_count"] to determine n_frames, and
    tests each trajectory by reading a single frame to ensure it is not corrupted.

    Returns:
        video: (3, T, 64, 64) float32 in [-1,1]
    """

    def __init__(self, data_roots, num_frames: int = 32, skip_initial_frames: int = 100):
        super().__init__()
        self.num_frames = num_frames
        self.skip_initial_frames = skip_initial_frames

        # Normalize to list
        if isinstance(data_roots, str):
            roots = [data_roots]
        else:
            roots = list(data_roots)

        self.index = []  # list of (dir_path, n_frames)

        total_trajs = 0
        usable_trajs = 0

        for root in roots:
            traj_dirs = sorted(
                [p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)]
            )

            print(f"Scanning directory {root} ({len(traj_dirs)} trajectories)...")

            for d in tqdm(traj_dirs, desc=f"Scanning {os.path.basename(root)}", unit="traj", leave=False):
                total_trajs += 1
                mp4_path = os.path.join(d, "recording.mp4")
                meta_path = os.path.join(d, "metadata.json")

                if not os.path.exists(mp4_path) or not os.path.exists(meta_path):
                    continue

                # 1) Load frame count from metadata.json
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    true_frames = int(meta.get("true_video_frame_count", 0))
                except Exception as e:
                    print(f"[SKIP] Failed to read metadata for {d}: {e}")
                    continue

                if true_frames <= 0:
                    # Nothing usable if frame count is missing or zero
                    continue

                n_frames = true_frames

                # 2) Check there are enough usable frames after skipping
                usable_frames = n_frames - self.skip_initial_frames
                if usable_frames < self.num_frames:
                    # Not enough frames after skipping artifacts
                    continue

                # 3) Check that the file is not obviously corrupted by reading ONE frame
                #    We pick a frame index that should be valid: e.g. at skip_initial_frames
                test_frame_idx = min(max(self.skip_initial_frames, 0), n_frames - 1)
                try:
                    reader = imageio.get_reader(mp4_path)
                    _ = reader.get_data(test_frame_idx)  # just one frame
                    reader.close()
                except Exception as e:
                    print(f"[SKIP] Corrupted video at {mp4_path}, frame {test_frame_idx}: {e}")
                    continue

                # If we got here, this trajectory is usable
                self.index.append((d, n_frames))
                usable_trajs += 1

        if not self.index:
            raise RuntimeError("No valid trajectories with enough frames found in data_roots")

        print(f"Found {usable_trajs}/{total_trajs} usable trajectories with >= {self.num_frames} frames.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dir_path, n_frames = self.index[idx]
        mp4_path = os.path.join(dir_path, "recording.mp4")

        # Valid start range is from skip_initial_frames to n_frames - num_frames
        start_min = self.skip_initial_frames
        start_max = n_frames - self.num_frames
        if start_max < start_min:
            # Should not happen if we filtered correctly
            raise IndexError(f"Not enough frames in {mp4_path} for requested window")

        # Random window inside valid range
        start = np.random.randint(start_min, start_max + 1)
        end = start + self.num_frames

        reader = imageio.get_reader(mp4_path)
        frames = []
        for t in range(start, end):
            frame = reader.get_data(t)  # (H,W,3)
            frames.append(frame)
        reader.close()

        video = np.stack(frames, axis=0)  # (T,H,W,3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0  # (3,T,H,W)
        video = video * 2.0 - 1.0  # [-1,1]
        return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more MineRL trajectory root folders",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_ckpt", type=str, default="tokenizer_ms_vqvae.pt")
    parser.add_argument("--base_channels", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--num_embeddings", type=int, default=1024)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    args = parser.parse_args()

    wandb.init(
        project="minecraft-tokenizer",
        config={
            "num_frames": args.num_frames,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "num_embeddings": args.num_embeddings,
            "epochs": args.epochs,
            "base_channels": args.base_channels,
        },
    )

    device = get_device()
    print("Using device:", device)

    dataset = MineRLVideoDataset(args.data_roots, num_frames=args.num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = Tokenizer(
        in_channels=3,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=0.02,
        num_res_blocks=args.num_res_blocks,
    ).to(device)

    describe_model(model)

    # Slightly higher LR to help the encoder learn; should still be stable on MPS
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # minimizing loss
        factor=0.8,        # new_lr = old_lr * 0.8
        patience=3,        # wait 3 epochs before reducing
    )

    best = float("inf")

    if os.path.exists(args.save_ckpt):
        print(f"Resuming from checkpoint {args.save_ckpt}")
        ckpt = torch.load(args.save_ckpt, map_location=device)

        model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt.get("epoch", 1)
    else:
        print("Starting fresh training")
        start_epoch = 1

    # Use tqdm for progress bar
    for epoch in trange(start_epoch, args.epochs + 1, desc="Training"):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        for _, vids in enumerate(loader):
            vids = vids.to(device)  # (B,3,T,64,64)

            tokens, vq_loss, z_q = model.encode_tokens(vids)
            # decode from quantized latents so gradients reach encoder/VQ
            recon = model.decode_latents(z_q)
            # keep outputs in range for stable loss
            recon = torch.tanh(recon)
            recon_loss = F.mse_loss(recon, vids)
            loss = recon_loss * 1.25 + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        avg = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_vq = total_vq / len(loader)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}: loss={avg:.6f}, recon={avg_recon:.6f}, vq={avg_vq:.6f}, lr={current_lr:.6e}")

        # Scheduler step
        scheduler.step(avg)

        wandb.log({
            "epoch": epoch,
            "loss": avg,
            "recon_loss": avg_recon,
            "vq_loss": avg_vq,
            "lr": current_lr,
        })

        if avg < best:
            best = avg
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                args.save_ckpt
            )
            print(f"New best ({best:.4f}), saved tokenizer to {args.save_ckpt}")
    torch.save(
        {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        args.save_ckpt
    )


if __name__ == "__main__":
    main()
