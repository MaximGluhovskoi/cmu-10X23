# train_var.py
import argparse
import os
from glob import glob

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer_ms_vqvae import MultiScaleVQTokenizerVideo2D as Tokenizer
from var_video import VARConfig, FrameConditionedVAR


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MineRLVideoActionDataset(Dataset):
    """
    For each trajectory, returns a num_frames clip and its actions.

    video:   (3, T, 64, 64) float32 in [-1,1]
    actions: (T-1, 11) float32
    """

    def __init__(self, data_root: str, num_frames: int = 32):
        super().__init__()
        self.num_frames = num_frames
        self.clip_dirs = sorted(
            [p for p in glob(os.path.join(data_root, "*")) if os.path.isdir(p)]
        )

        self.index = []
        for d in self.clip_dirs:
            mp4_path = os.path.join(d, "recording.mp4")
            npz_path = os.path.join(d, "rendered.npz")
            if not (os.path.exists(mp4_path) and os.path.exists(npz_path)):
                continue

            arr = np.load(npz_path)
            T_actions = arr["action$forward"].shape[0]

            reader = imageio.get_reader(mp4_path)
            n_frames = reader.count_frames()
            reader.close()

            # we can only use up to actions+1 frames
            T_frames = min(n_frames, T_actions + 1)
            if T_frames < num_frames:
                continue

            self.index.append((d, T_frames))

        if not self.index:
            raise RuntimeError("No valid trajectories found in data_root")

    def __len__(self):
        return len(self.index)

    def _load_actions(self, npz_path, start, length):
        """Load actions as float32 array of shape (length-1, 11)."""
        arr = np.load(npz_path)
        end = start + length - 1  # actions are between frames

        forward = arr["action$forward"][start:end]
        left = arr["action$left"][start:end]
        back = arr["action$back"][start:end]
        right = arr["action$right"][start:end]
        jump = arr["action$jump"][start:end]
        sneak = arr["action$sneak"][start:end]
        sprint = arr["action$sprint"][start:end]
        attack = arr["action$attack"][start:end]
        place_raw = arr["action$place"][start:end]
        camera = arr["action$camera"][start:end]

        # handle string "none" for place
        if place_raw.dtype.kind in {"U", "S", "O"}:
            place = (place_raw != "none").astype(np.float32)
        else:
            place = place_raw.astype(np.float32)

        scalars = np.stack(
            [forward, left, back, right, jump, sneak, sprint, attack, place],
            axis=-1,
        ).astype(np.float32)

        acts = np.concatenate([scalars, camera.astype(np.float32)], axis=-1)  # (L-1, 11)
        return acts

    def __getitem__(self, idx):
        dir_path, T_frames = self.index[idx]
        mp4_path = os.path.join(dir_path, "recording.mp4")
        npz_path = os.path.join(dir_path, "rendered.npz")

        # sample a random window to expose the model to different parts of the trajectory
        start = np.random.randint(0, T_frames - self.num_frames + 1)
        end = start + self.num_frames

        # load frames
        reader = imageio.get_reader(mp4_path)
        frames = []
        for t in range(start, end):
            frame = reader.get_data(t)  # (64,64,3)
            frames.append(frame)
        reader.close()

        video = np.stack(frames, axis=0)  # (T,H,W,3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0  # (3,T,H,W)
        video = video * 2.0 - 1.0  # [-1,1]

        # load actions for T-1 transitions
        actions = self._load_actions(npz_path, start, self.num_frames)
        actions = torch.from_numpy(actions).float()  # (T-1, 11)

        return video, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--save_ckpt", type=str, default="var_video.pt")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    # dataset
    dataset = MineRLVideoActionDataset(args.data_root, num_frames=args.num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # tokenizer – MUST match train_tokenizer.py
    tokenizer = Tokenizer(
        in_channels=3,
        base_channels=64,
        latent_dim=128,      # <- match tokenizer_ms_vqvae.py
        num_embeddings=256,  # <- match tokenizer_ms_vqvae.py
        commitment_cost=0.02,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    # VAR model – codebook_size must match num_embeddings (256)
    cfg = VARConfig()  # assuming var_video.py has codebook_size=256
    var_model = FrameConditionedVAR(cfg).to(device)

    optimizer = torch.optim.Adam(var_model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        var_model.train()
        total_loss = 0.0

        for vids, acts in loader:
            vids = vids.to(device)  # (B,3,T,64,64)
            acts = acts.to(device)  # (B,T-1,11)
            B, _, T, H, W = vids.shape

            # freeze tokenizer
            with torch.no_grad():
                tokens_list, _, _ = tokenizer.encode_tokens(vids)
            tokens = tokens_list[0]  # (B,T,H_tok,W_tok)
            B, T_tok, H_tok, W_tok = tokens.shape

            loss = 0.0
            count = 0

            # one-step prediction: p(F_t | F_{t-1}, a_{t-1})
            for t in range(1, T_tok):
                prev_tok = tokens[:, t - 1]   # (B,H_tok,W_tok)
                tgt_tok = tokens[:, t]        # (B,H_tok,W_tok)
                act_prev = acts[:, t - 1]     # (B,11)

                logits = var_model(prev_tok, tgt_tok, act_prev)  # (B,H_tok,W_tok,V)
                loss_t = F.cross_entropy(
                    logits.view(-1, cfg.codebook_size),
                    tgt_tok.view(-1),
                )
                loss = loss + loss_t
                count += 1

            loss = loss / max(count, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg:.4f}")
        torch.save(var_model.state_dict(), args.save_ckpt)
        print(f"Saved VAR checkpoint to {args.save_ckpt}")


if __name__ == "__main__":
    main()
