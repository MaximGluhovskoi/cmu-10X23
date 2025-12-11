# train_var.py
import argparse
import os
from glob import glob
import math

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

from tokenizer_ms_vqvae import MultiScaleVQTokenizerVideo2D as Tokenizer
from var_video import VARConfig, FrameConditionedVAR

import wandb



def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MineRLVideoActionDataset(Dataset):
    """
    For each trajectory folder under data_root, this dataset reads:
      - recording.mp4
      - rendered.npz

    For each __getitem__, it returns a random window of length num_frames:

        video:   (3, T, 64, 64) float32 in [-1,1]
        actions: (T-1, 11) float32

    The actions are:
        [forward, left, back, right,
         jump, sneak, sprint, attack,
         place, camera_x, camera_y]
    """

    def __init__(self, data_roots: str, num_frames: int = 32):
        super().__init__()
        self.num_frames = num_frames
        # Allow passing a single string or a list of strings
        if isinstance(data_roots, str):
            data_roots = [data_roots]

        self.clip_dirs = []
        for root in data_roots:
            self.clip_dirs.extend(
                [p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)]
            )
        self.clip_dirs = sorted(self.clip_dirs)

        self.index = []  # list of (dir_path, T_frames)
        for d in self.clip_dirs:
            mp4_path = os.path.join(d, "recording.mp4")
            npz_path = os.path.join(d, "rendered.npz")
            if not (os.path.exists(mp4_path) and os.path.exists(npz_path)):
                continue

            arr = np.load(npz_path)
            # number of action steps available
            T_actions = arr["action$forward"].shape[0]

            # number of frames in the video
            reader = imageio.get_reader(mp4_path)
            n_frames = reader.get_length()
            reader.close()

            # actions are between frames, so at most actions+1 frames are aligned
            T_frames = min(n_frames, T_actions + 1)
            if T_frames < num_frames:
                continue

            self.index.append((d, T_frames))

        if not self.index:
            raise RuntimeError(
                "No valid trajectories with enough frames & actions found in data_root"
            )

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _load_actions(npz_path: str, start: int, length: int) -> np.ndarray:
        """
        Load actions for a window of length frames, i.e. (length-1) transitions.

        Returns:
            acts: (length-1, 11) float32
        """
        arr = np.load(npz_path)
        end = start + length - 1  # actions live between frames

        forward = arr["action$forward"][start:end]
        left = arr["action$left"][start:end]
        back = arr["action$back"][start:end]
        right = arr["action$right"][start:end]
        jump = arr["action$jump"][start:end]
        sneak = arr["action$sneak"][start:end]
        sprint = arr["action$sprint"][start:end]
        attack = arr["action$attack"][start:end]
        place_raw = arr["action$place"][start:end]
        camera = arr["action$camera"][start:end]  # typically (L-1, 2)

        # handle string "none" for place
        if place_raw.dtype.kind in {"U", "S", "O"}:
            place = (place_raw != "none").astype(np.float32)
        else:
            place = place_raw.astype(np.float32)

        scalars = np.stack(
            [forward, left, back, right, jump, sneak, sprint, attack, place],
            axis=-1,
        ).astype(np.float32)

        camera = camera.astype(np.float32)
        if camera.ndim == 1:
            # edge-case: some logs might store camera as scalar; expand to (L-1, 1)
            camera = camera[:, None]

        acts = np.concatenate([scalars, camera], axis=-1)  # (L-1, 11)
        return acts

    def __getitem__(self, idx):
        dir_path, T_frames = self.index[idx]
        mp4_path = os.path.join(dir_path, "recording.mp4")
        npz_path = os.path.join(dir_path, "rendered.npz")

        SKIP = 100  # skip first 100 frames

        valid_start_max = T_frames - self.num_frames
        start = np.random.randint(SKIP, valid_start_max + 1)
        end = start + self.num_frames

        # load frames
        reader = imageio.get_reader(mp4_path)
        frames = []
        for t in range(start, end):
            frame = reader.get_data(t)  # (64,64,3) uint8
            frames.append(frame)
        reader.close()

        video = np.stack(frames, axis=0)  # (T,H,W,3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0  # (3,T,H,W)
        video = video * 2.0 - 1.0  # [-1,1]

        # load actions for T-1 transitions
        actions = self._load_actions(npz_path, start, self.num_frames)
        actions = torch.from_numpy(actions).float()  # (T-1, 11)

        return video, actions


# --------------------------------------------------------------------------------------
# Multi-scale VAR for images (next-scale prediction)
# --------------------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, D)
        attn_mask: optional (B, N, N) with 0 for allowed, -inf for blocked
        """
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B,N,3D)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(1)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,N,head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B,N,D)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class VARImageConfig:
    """
    Config for multi-scale VAR over 16x16 VQ-VAE tokens.
    """
    codebook_size: int = 256   # must match tokenizer num_embeddings
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_h: int = 16            # base latent height (e.g., 16 for 64x64 images)
    max_w: int = 16
    action_dim: int = 11       # keyboard + mouse actions
    dropout: float = 0.1


def compute_multiscale_tokens(tokens_frame: torch.Tensor):
    """
    Given a single-frame latent grid of shape (B, H, W), build a list of
    coarse-to-fine discrete token grids using simple strided subsampling.

    For H=W=16, this yields:
        [(B,1,1), (B,2,2), (B,4,4), (B,8,8), (B,16,16)]
    """
    B, H, W = tokens_frame.shape
    assert H == W, "Only square latent grids are supported for now."

    # build strides from coarsest (full image) down to finest (1 px)
    strides = []
    s = H
    while s >= 1:
        strides.append(s)
        s //= 2
    # strides: [16, 8, 4, 2, 1] for H=16

    scales = []
    for stride in reversed(strides):
        # e.g., stride=16 -> (B,1,1), stride=8 -> (B,2,2), ...
        scales.append(tokens_frame[:, ::stride, ::stride])

    return scales  # coarse -> fine


class MultiScaleVAR(nn.Module):
    """
    Multi-scale Visual Autoregressive model.

    For a given frame's latent tokens z (16x16), we derive a pyramid:
        z^0 (1x1), z^1 (2x2), ..., z^L (16x16).

    The model factorizes:
        p(z^0, z^1, ..., z^L | a) = Π_s p(z^s | z^{<s}, a)

    At each scale s, it conditions on all coarser tokens (0..s-1) and
    an optional action embedding a (keyboard+mouse), and predicts z^s.
    """

    def __init__(self, cfg: VARImageConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.codebook_size, cfg.d_model)
        self.row_embed = nn.Embedding(cfg.max_h, cfg.d_model)
        self.col_embed = nn.Embedding(cfg.max_w, cfg.d_model)

        # max number of scales for a max_h x max_w grid
        self.num_scales = int(math.log2(cfg.max_h)) + 1
        self.scale_embed = nn.Embedding(self.num_scales, cfg.d_model)

        self.query_token = nn.Parameter(torch.randn(cfg.d_model))

        # optional action conditioning
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)
        self.action_token = nn.Parameter(torch.randn(cfg.d_model))

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
             for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.codebook_size)

    def _base_hw(self, tokens_scales):
        # we assume the finest scale is last
        _, H_base, W_base = tokens_scales[-1].shape
        return H_base, W_base

    def _coords_for_scale(
        self,
        H_base: int,
        W_base: int,
        H_s: int,
        W_s: int,
        device: torch.device,
    ):
        """
        Build global row/col indices (in 0..H_base-1 / 0..W_base-1) for a
        scale with shape (H_s, W_s) using uniform strided sampling.
        """
        stride_h = H_base // H_s
        stride_w = W_base // W_s
        rows = torch.arange(0, H_base, stride_h, device=device)[:H_s]
        cols = torch.arange(0, W_base, stride_w, device=device)[:W_s]
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        return row_grid, col_grid

    def forward(
        self,
        tokens_scales,
        action: torch.Tensor = None,
    ):
        """
        tokens_scales: list of length S, each (B, H_s, W_s), coarse->fine.
        action: optional (B, A) vector to condition on (same for all scales).

        Returns:
            logits_scales: list of length S,
                each (B, H_s, W_s, V) with logits for that scale.
        """
        cfg = self.cfg
        device = tokens_scales[0].device
        B = tokens_scales[0].shape[0]
        H_base, W_base = self._base_hw(tokens_scales)

        logits_scales = []

        # Precompute action embedding if provided
        if action is not None:
            act_emb = self.action_proj(action) + self.action_token  # (B,D)
            act_emb = act_emb.unsqueeze(1)  # (B,1,D)
        else:
            act_emb = None

        S = len(tokens_scales)
        assert S <= self.num_scales, f"Got {S} scales but model only supports {self.num_scales}"

        for s in range(S):
            # Build context from all coarser scales 0..s-1
            ctx_ids = []
            ctx_row = []
            ctx_col = []
            ctx_scale = []

            for k in range(s):
                tok_k = tokens_scales[k]  # (B,Hk,Wk)
                _, Hk, Wk = tok_k.shape
                row_grid, col_grid = self._coords_for_scale(H_base, W_base, Hk, Wk, device)
                row_flat = row_grid.reshape(1, -1).expand(B, -1)
                col_flat = col_grid.reshape(1, -1).expand(B, -1)
                ids_flat = tok_k.view(B, -1)
                scale_flat = torch.full_like(ids_flat, fill_value=k)

                ctx_ids.append(ids_flat)
                ctx_row.append(row_flat)
                ctx_col.append(col_flat)
                ctx_scale.append(scale_flat)

            if ctx_ids:
                ctx_ids = torch.cat(ctx_ids, dim=1)
                ctx_row = torch.cat(ctx_row, dim=1)
                ctx_col = torch.cat(ctx_col, dim=1)
                ctx_scale = torch.cat(ctx_scale, dim=1)

                ctx_emb = self.token_embed(ctx_ids)
                ctx_emb = (
                    ctx_emb
                    + self.row_embed(ctx_row.clamp(max=cfg.max_h - 1))
                    + self.col_embed(ctx_col.clamp(max=cfg.max_w - 1))
                    + self.scale_embed(ctx_scale.clamp(max=self.num_scales - 1))
                )
            else:
                ctx_emb = None

            # Build query tokens for current scale s
            tok_s = tokens_scales[s]
            _, Hs, Ws = tok_s.shape
            row_grid_s, col_grid_s = self._coords_for_scale(H_base, W_base, Hs, Ws, device)
            row_q = row_grid_s.reshape(1, -1).expand(B, -1)
            col_q = col_grid_s.reshape(1, -1).expand(B, -1)
            scale_q = torch.full_like(row_q, fill_value=s)

            L_q = Hs * Ws
            query_vec = self.query_token.view(1, 1, -1).expand(B, L_q, -1)
            query_emb = (
                query_vec
                + self.row_embed(row_q.clamp(max=cfg.max_h - 1))
                + self.col_embed(col_q.clamp(max=cfg.max_w - 1))
                + self.scale_embed(scale_q.clamp(max=self.num_scales - 1))
            )

            # Assemble sequence: [context tokens] + [query tokens] (+ optional action token)
            pieces = []
            if ctx_emb is not None:
                pieces.append(ctx_emb)
            pieces.append(query_emb)
            if act_emb is not None:
                pieces.append(act_emb)

            x = torch.cat(pieces, dim=1)  # (B, L_ctx + L_q (+1), D)

            # No additional mask needed: queries only see context+itself
            # because we never include finer-scale tokens in ctx.
            attn_mask = None

            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)

            x = self.norm(x)

            # queries occupy the block immediately after context tokens
            offset = ctx_emb.shape[1] if ctx_emb is not None else 0
            query_out = x[:, offset : offset + L_q, :]  # (B,L_q,D)
            logits_s = self.head(query_out).view(B, Hs, Ws, cfg.codebook_size)
            logits_scales.append(logits_s)

        return logits_scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more MineRL trajectory root folders",
    )
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--save_ckpt", type=str, default="var_video.pt")
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    # tokenizer hyperparams (must match train_tokenizer.py)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=256)
    parser.add_argument("--num_res_blocks", type=int, default=0)

    # VAR hyperparams
    parser.add_argument("--var_d_model", type=int, default=256)
    parser.add_argument("--var_n_heads", type=int, default=4)
    parser.add_argument("--var_n_layers", type=int, default=4)
    parser.add_argument("--var_dropout", type=float, default=0.1)

    args = parser.parse_args()

    wandb.init(
        project="minecraft-var-video",
        config={
            "num_frames": args.num_frames,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "var_d_model": args.var_d_model,
            "var_n_heads": args.var_n_heads,
            "var_n_layers": args.var_n_layers,
            "var_dropout": args.var_dropout,
            "lr": 1e-4,
            "num_embeddings": args.num_embeddings,
        },
    )

    device = get_device()
    print("Using device:", device)

    # dataset
    dataset = MineRLVideoActionDataset(args.data_roots, num_frames=args.num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # tokenizer – MUST match train_tokenizer.py
    tokenizer = Tokenizer(
        in_channels=3,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=0.02,
        num_res_blocks=args.num_res_blocks,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # VAR model – multi-scale next-scale image generator
    cfg = VARImageConfig(
        codebook_size=args.num_embeddings,
        d_model=args.var_d_model,
        n_heads=args.var_n_heads,
        n_layers=args.var_n_layers,
        max_h=16,
        max_w=16,
        action_dim=11,
        dropout=args.var_dropout,
    )
    var_model = MultiScaleVAR(cfg).to(device)

    optimizer = torch.optim.Adam(var_model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # minimizing loss
        factor=0.8,        # new_lr = old_lr * 0.8
        patience=5,        # wait 3 epochs before reducing
    )

    if os.path.exists(args.save_ckpt):
        print(f"Resuming from checkpoint {args.save_ckpt}")
        ckpt = torch.load(args.save_ckpt, map_location=device)

        var_model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt.get("epoch", 1)
    else:
        print("Starting fresh training")
        start_epoch = 1

    best = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        var_model.train()
        total_loss = 0.0
        n_batches = 0

        for vids, acts in loader:
            vids = vids.to(device)  # (B,3,T,64,64)
            acts = acts.to(device)  # (B,T-1,11)
            B, _, T, H, W = vids.shape

            with torch.no_grad():
                tokens_list, _, _ = tokenizer.encode_tokens(vids)
            tokens = tokens_list[0]  # (B,T,H_tok,W_tok)
            _, T_tok, H_tok, W_tok = tokens.shape

            # sanity check: H_tok/W_tok should match cfg.max_h/max_w
            if (H_tok != cfg.max_h) or (W_tok != cfg.max_w):
                raise RuntimeError(
                    f"Tokenizer latent shape {(H_tok, W_tok)} does not match "
                    f"VAR config max_h/max_w={(cfg.max_h, cfg.max_w)}"
                )

            loss = 0.0
            count = 0

            # train on each frame t >= 1 with previous action a_{t-1} as condition
            for t in range(1, T_tok):
                frame_tokens = tokens[:, t]       # (B,H_tok,W_tok)
                act_prev = acts[:, t - 1]        # (B,11)

                tokens_scales = compute_multiscale_tokens(frame_tokens)
                logits_scales = var_model(tokens_scales, action=act_prev)

                # multi-scale cross-entropy loss
                loss_t = 0.0
                for tok_s, logit_s in zip(tokens_scales, logits_scales):
                    loss_s = F.cross_entropy(
                        logit_s.view(-1, cfg.codebook_size),
                        tok_s.view(-1),
                    )
                    loss_t = loss_t + loss_s

                loss_t = loss_t / len(tokens_scales)
                loss = loss + loss_t
                count += 1

            loss = loss / max(count, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        token_perplexity = math.exp(avg)

        wandb.log(
            {
                "epoch": epoch,
                "loss": avg,
                "learning_rate": current_lr,
                "token_perplexity": token_perplexity,
            }
        )

        # Scheduler step
        scheduler.step(avg)

        print(f"Epoch {epoch}: loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": var_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                args.save_ckpt
            )
            print(f"Saved VAR checkpoint to {args.save_ckpt}")


if __name__ == "__main__":
    main()
