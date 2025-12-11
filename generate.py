"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import load_prompt, load_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os
import time
import numpy as np

# Optional metrics imports
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not found. Install with: pip install lpips to compute t-LPIPS metric.")

# adjusting code for cuda or mps
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    raise RuntimeError("No cuda or mps device available!")


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = DiT_models["DiT-S/2"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith(".pt"):
        ckpt = torch.load(args.oasis_ckpt, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
    if args.vae_ckpt.endswith(".pt"):
        vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif args.vae_ckpt.endswith(".safetensors"):
        load_model(vae, args.vae_ckpt)
    vae = vae.to(device).eval()

    # sampling params
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    # get prompt image/video
    x = load_prompt(
        args.prompt_path,
        video_offset=args.video_offset,
        n_prompt_frames=n_prompt_frames,
    )
    # get input action stream
    actions = load_actions(args.actions_path, action_offset=args.video_offset)[:, :total_frames]

    # sampling inputs
    x = x.to(device)
    actions = actions.to(device)

    # vae encoding
    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        # Use MPS if available, otherwise CUDA, fallback to no autocast
        if device == "mps":
            # MPS doesn't support autocast the same way, but we can still use it
            x = vae.encode(x * 2 - 1).mean * scaling_factor
        elif device.startswith("cuda"):
            with autocast("cuda", dtype=torch.half):
                x = vae.encode(x * 2 - 1).mean * scaling_factor
        else:
            x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
    x = x[:, :n_prompt_frames]

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # Start timing inference
    inference_start_time = time.time()

    # sampling loop
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # set up noise values
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            # get model predictions
            with torch.no_grad():
                if device.startswith("cuda"):
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])
                else:
                    # MPS or CPU - no autocast needed
                    v = model(x_curr, t, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    # vae decoding
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        if device.startswith("cuda"):
            with autocast("cuda", dtype=torch.half):
                x = (vae.decode(x / scaling_factor) + 1) / 2
        else:
            x = (vae.decode(x / scaling_factor) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    # End timing inference
    inference_time = time.time() - inference_start_time
    num_generated_frames = total_frames - n_prompt_frames

    # save video
    x = torch.clamp(x, 0, 1)
    x_uint8 = (x * 255).byte()
    write_video(args.output_path, x_uint8[0].cpu(), fps=args.fps)
    print(f"generation saved to {args.output_path}.")

    # Compute metrics
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"Total inference time: {inference_time:.2f}s")
    print(f"Time per frame: {inference_time / num_generated_frames:.4f}s")
    print(f"Number of generated frames: {num_generated_frames}")

    # Compute temporal LPIPS (t-LPIPS) if available
    if LPIPS_AVAILABLE and args.compute_metrics:
        print("\nComputing temporal consistency (t-LPIPS)...")
        try:
            # Handle SSL certificate issues for downloading weights
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            lpips_model = lpips.LPIPS(net='alex').to(device)
            lpips_model.eval()
            
            # Extract generated frames (skip prompt frames)
            generated_frames = x[0, n_prompt_frames:]  # (T, H, W, C)
            
            # Convert to (T, C, H, W) for LPIPS
            frames_lpips = rearrange(generated_frames, "t h w c -> t c h w")
            frames_lpips = frames_lpips * 2.0 - 1.0  # Convert to [-1, 1]
            frames_lpips = frames_lpips.to(device)
            
            temporal_dists = []
            for t in range(frames_lpips.shape[0] - 1):
                frame_t = frames_lpips[t:t+1]  # (1, C, H, W)
                frame_t1 = frames_lpips[t+1:t+2]  # (1, C, H, W)
                
                with torch.no_grad():
                    dist = lpips_model(frame_t, frame_t1)
                temporal_dists.append(dist.item())
            
            t_lpips = np.mean(temporal_dists)
            print(f"t-LPIPS (temporal consistency): {t_lpips:.4f} (lower is better)")
            
        except Exception as e:
            print(f"Warning: Could not compute t-LPIPS: {e}")
            t_lpips = None
    else:
        t_lpips = None
        if not LPIPS_AVAILABLE:
            print("\nSkipping t-LPIPS (lpips not installed)")
        elif not args.compute_metrics:
            print("\nSkipping t-LPIPS (use --compute-metrics to enable)")

    # Save metrics to file if requested
    if args.output_metrics:
        import json
        metrics = {
            "inference_time_total": inference_time,
            "inference_time_per_frame": inference_time / num_generated_frames,
            "num_generated_frames": num_generated_frames,
            "num_total_frames": total_frames,
            "num_prompt_frames": n_prompt_frames,
            "t_lpips": float(t_lpips) if t_lpips is not None else None,
        }
        with open(args.output_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output_metrics}")
    
    print("="*50)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="oasis500m.safetensors",
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default="vit-l-20.safetensors",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=32,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="sample_data/sample_image_0.png",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="sample_data/sample_actions_0.one_hot_actions.pt",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=None,
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=1,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="video.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=10)
    parse.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute performance metrics (t-LPIPS, inference time)",
    )
    parse.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save metrics JSON file",
    )

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
