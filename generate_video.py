import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video
import imageio.v2 as imageio

from vae import VAE_models


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


import imageio.v2 as imageio
import torch
import torch.nn.functional as F

@torch.no_grad()
def load_video_frames(video_path, target_h=360, target_w=640, max_frames=240):
    """
    Stream video with imageio and only keep up to max_frames.
    Returns:
        frames: (T, 3, H, W) in [0,1] on CPU
        fps: frames per second
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30)

    frames_list = []

    for i, frame in enumerate(reader):
        print(f"Reading frame {i}", end="\r")
        if i >= max_frames:
            break

        # frame: (H, W, C), uint8
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        # resize to (target_h, target_w)
        frame_tensor = F.interpolate(
            frame_tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (3, target_h, target_w)

        frames_list.append(frame_tensor)

    reader.close()

    if len(frames_list) == 0:
        raise RuntimeError("No frames read from video")

    frames = torch.stack(frames_list, dim=0)  # (T, 3, H, W)
    return frames, fps



@torch.no_grad()
def run_autoencoder_on_video(
    model,
    frames,
    batch_size=8,
):
    """
    frames: (T, 3, H, W) tensor in [0,1]
    Returns:
        recon_frames: (T, 3, H, W) reconstruction tensor
        latents_list: list of latent tensors per batch (optional)
    """
    device = next(model.parameters()).device
    T = frames.shape[0]

    all_recons = []
    all_latents = []

    from tqdm import trange   # or tqdm
    print("Running autoencoder on video frames...")
    for start in trange(0, T, batch_size, desc="Autoencoding"):
        end = min(start + batch_size, T)
        batch = frames[start:end]  # (B, 3, H, W)
        batch = batch.to(device)

        # AutoencoderKL.forward takes (inputs, labels, split="train")
        # but we do not need labels here, so pass None
        rec, posterior, latent = model(batch, labels=None, split="val")

        all_recons.append(rec.cpu())
        all_latents.append(latent.cpu())

    recon_frames = torch.cat(all_recons, dim=0)  # (T, 3, H, W)
    latents = torch.cat(all_latents, dim=0)      # (T, seq_len, latent_dim)

    return recon_frames, latents


@torch.no_grad()
def save_video(frames, out_path, fps=30):
    """
    frames: (T, 3, H, W) in [0,1] tensor
    Saves as an mp4 using imageio.
    """
    frames = frames.clamp(0.0, 1.0)
    frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)  # (T, H, W, C)

    imageio.mimwrite(out_path, list(frames_np), fps=fps)


@torch.no_grad()
def save_side_by_side(orig_frames, recon_frames, out_path, fps=30):
    """
    orig_frames, recon_frames: (T, 3, H, W) in [0,1]
    Writes a side by side comparison video.
    """
    orig = orig_frames.clamp(0.0, 1.0)
    recon = recon_frames.clamp(0.0, 1.0)

    orig_np = (orig.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)   # (T, H, W, C)
    recon_np = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8) # (T, H, W, C)

    T, H, W, C = orig_np.shape
    canvas_list = []
    for t in range(T):
        canvas = np.zeros((H, W * 2, C), dtype=np.uint8)
        canvas[:, :W] = orig_np[t]
        canvas[:, W:] = recon_np[t]
        canvas_list.append(canvas)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimwrite(out_path, canvas_list, fps=fps)


def load_model(model_name, ckpt_path, device):
    """
    Instantiate and load the VAE model from a checkpoint.
    """
    if model_name not in VAE_models:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(VAE_models.keys())}")

    model_fn = VAE_models[model_name]
    model = model_fn()  # you can pass kwargs here if you want to override defaults
    model.to(device)

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        # adjust key here if your checkpoint structure is different
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint from {ckpt_path}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input mp4")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VAE checkpoint (.pth or .pt)")
    parser.add_argument("--out", type=str, help="Path to output mp4 for recon or side by side", default="output3.mp4")
    parser.add_argument("--model_name", type=str, default="vit-l-20-shallow-encoder")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--side_by_side", action="store_true", help="Save side by side comparison video")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # 1. Load at most 240 frames on CPU
    frames, fps = load_video_frames(
        args.video,
        target_h=360,
        target_w=640,
        max_frames=1024,
    )
    print(f"Loaded {frames.shape[0]} frames at {fps:.2f} fps")
    if device is not None:
        frames = frames.to(device)

    print(f"Loaded video: {frames.shape[0]} frames at {fps:.2f} fps")

    # 2. Load model
    model = load_model(args.model_name, args.ckpt, device)

    # 3. Run autoencoder on video
    recon_frames, latents = run_autoencoder_on_video(
        model=model,
        frames=frames,
        batch_size=args.batch_size,
    )
    print(f"Reconstructed frames shape: {recon_frames.shape}")
    print(f"Latent shape: {latents.shape}")

    # 4. Save output video
    if args.side_by_side:
        save_side_by_side(frames.cpu(), recon_frames.cpu(), args.out, fps=fps)
        print(f"Saved side by side video to {args.out}")
    else:
        save_video(recon_frames.cpu(), args.out, fps=fps)
        print(f"Saved reconstructed video to {args.out}")


if __name__ == "__main__":
    main()
