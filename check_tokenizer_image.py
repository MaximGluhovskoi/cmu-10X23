import argparse
import os
import numpy as np
from PIL import Image
import torch
import imageio.v2 as imageio

from tokenizer_ms_vqvae import MultiScaleVQTokenizerVideo2D as Tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_side_by_side(orig, recon, out_path):
    # orig, recon: (3, H, W) in [0,1]
    orig_img = (orig * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    recon_img = (recon * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

    h, w, _ = orig_img.shape
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = orig_img
    canvas[:, w:] = recon_img

    Image.fromarray(canvas).save(out_path)
    print(f"Saved comparison: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="Path to MineRL trajectory folder containing recording.mp4")
    parser.add_argument("--frame_idx", type=int, default=900,
                        help="Frame to analyze (default 200)")
    parser.add_argument("--out_image", type=str, default="tokenizer_check_video.png")

    # Must match tokenizer training
    parser.add_argument("--base_channels", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--num_embeddings", type=int, default=1024)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    mp4_path = os.path.join(args.traj_dir, "recording.mp4")
    assert os.path.exists(mp4_path), f"Could not find {mp4_path}"

    reader = imageio.get_reader(mp4_path)
    num_frames = reader.get_length()

    print(f"Total frames in video: {num_frames}")

    frame_idx = min(args.frame_idx, num_frames - 1)
    frame_np = reader.get_data(frame_idx)  # (H,W,3)
    reader.close()

    # Resize to 64x64
    img = Image.fromarray(frame_np).convert("RGB").resize((64, 64), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0  # (64,64,3)
    orig_for_save = img_np.transpose(2, 0, 1)  # (3,64,64) in [0,1]

    # normalize to [-1,1] and wrap into (1,3,1,64,64)
    img_t = torch.from_numpy(orig_for_save).unsqueeze(0).unsqueeze(2)  # (1,3,1,64,64)
    img_t = img_t * 2.0 - 1.0
    img_t = img_t.to(device)

    # Sanity check: invert the transforms applied to img_t
    # img_t was: x in [0,1] -> x*2-1, so inverse is (x+1)/2
    with torch.no_grad():
        orig_from_tensor = ((img_t[0, :, 0] + 1.0) * 0.5).cpu().numpy()  # (3,64,64) in [0,1]

    # Load tokenizer
    tokenizer = Tokenizer(
        in_channels=3,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=0.02,
        num_res_blocks=args.num_res_blocks,
    ).to(device)

    ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        tokenizer.load_state_dict(ckpt["model"])
    else:
        tokenizer.load_state_dict(ckpt)

    tokenizer.eval()

    # Encode and decode
    with torch.no_grad():
        tokens_list, vq_loss, z_q = tokenizer.encode_tokens(img_t)
        recon = tokenizer.decode_latents(z_q)
        # model outputs in [-1,1]; map back to [0,1]
        recon_np = ((recon[0, :, 0] + 1.0) * 0.5).cpu().numpy()

    mse = np.mean((orig_from_tensor - recon_np) ** 2)
    print("Reconstruction MSE for this frame:", mse)

    # Save side by side: left = inverse of img_t, right = recon
    save_side_by_side(orig_from_tensor, recon_np, args.out_image)


if __name__ == "__main__":
    main()
