# generate_var.py
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from tokenizer_ms_vqvae import MultiScaleVQTokenizerVideo2D as Tokenizer
from var_video import VARConfig, FrameConditionedVAR


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_video_and_actions(traj_dir, num_frames):
    mp4_path = os.path.join(traj_dir, "recording.mp4")
    npz_path = os.path.join(traj_dir, "rendered.npz")

    if not (os.path.exists(mp4_path) and os.path.exists(npz_path)):
        raise FileNotFoundError("Missing recording.mp4 or rendered.npz in trajectory folder")

    arr = np.load(npz_path)
    T_actions = arr["action$forward"].shape[0]

    reader = imageio.get_reader(mp4_path)
    n_frames = reader.count_frames()
    reader.close()

    T_frames = min(n_frames, T_actions + 1)
    if T_frames < num_frames:
        raise ValueError(f"Trajectory has only {T_frames} frames, need {num_frames}")

    start = 0
    end = start + num_frames

    # load frames
    reader = imageio.get_reader(mp4_path)
    frames = []
    for t in range(start, end):
        frame = reader.get_data(t)  # (64,64,3)
        frames.append(frame)
    reader.close()

    video = np.stack(frames, axis=0)  # (T,H,W,3)
    video_t = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0  # (3,T,H,W)
    video_t = video_t * 2.0 - 1.0  # [-1,1]

    # load actions for T-1 transitions
    forward = arr["action$forward"][start:end - 1]
    left = arr["action$left"][start:end - 1]
    back = arr["action$back"][start:end - 1]
    right = arr["action$right"][start:end - 1]
    jump = arr["action$jump"][start:end - 1]
    sneak = arr["action$sneak"][start:end - 1]
    sprint = arr["action$sprint"][start:end - 1]
    attack = arr["action$attack"][start:end - 1]
    place_raw = arr["action$place"][start:end - 1]
    camera = arr["action$camera"][start:end - 1]

    # handle string "none" for place
    if place_raw.dtype.kind in {"U", "S", "O"}:
        place = (place_raw != "none").astype(np.float32)
    else:
        place = place_raw.astype(np.float32)

    scalars = np.stack(
        [forward, left, back, right, jump, sneak, sprint, attack, place],
        axis=-1,
    ).astype(np.float32)

    acts = np.concatenate([scalars, camera.astype(np.float32)], axis=-1)  # (T-1,11)
    acts_t = torch.from_numpy(acts).float()

    # (1,3,T,64,64), (1,T-1,11)
    return video_t.unsqueeze(0), acts_t.unsqueeze(0)


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, -1e10), logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--var_ckpt", type=str, required=False,
                        help="VAR checkpoint (not needed if --teacher_forced)")
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--out_video", type=str, default="var_generated.mp4")
    parser.add_argument("--teacher_forced", action="store_true",
                        help="Bypass VAR: encode+decode real video to check tokenizer quality")
    # tokenizer hyperparams (must match training)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=256)
    parser.add_argument("--num_res_blocks", type=int, default=0)
    # VAR hyperparams (must match training)
    parser.add_argument("--var_d_model", type=int, default=256)
    parser.add_argument("--var_n_heads", type=int, default=4)
    parser.add_argument("--var_n_layers", type=int, default=4)
    parser.add_argument("--var_dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    # tokenizer – MUST match train_tokenizer.py
    tokenizer = Tokenizer(
        in_channels=3,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,      # match tokenizer training
        num_embeddings=args.num_embeddings,  # match tokenizer training
        commitment_cost=0.02,
        num_res_blocks=args.num_res_blocks,
    ).to(device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=device))
    tokenizer.eval()

    video, actions = load_video_and_actions(args.traj_dir, args.num_frames)
    video = video.to(device)      # (1,3,T,64,64)
    actions = actions.to(device)  # (1,T-1,11)
    T = video.shape[2]

    # Fast sanity check: if tokenizer recon is bad, VAR cannot fix it
    if args.teacher_forced:
        with torch.no_grad():
            tokens_gt, _, z_q = tokenizer.encode_tokens(video)
            recon = tokenizer.decode_latents(z_q)
            recon = torch.tanh(recon)
            recon = recon.clamp(-1.0, 1.0)
            recon = (recon[0] * 0.5 + 0.5).cpu().numpy()  # (3,T,64,64)
            recon = (recon * 255.0).astype(np.uint8)
        C, T_recon, H, W = recon.shape
        writer = imageio.get_writer(args.out_video, fps=20)
        for t in range(T_recon):
            frame = recon[:, t].transpose(1, 2, 0)  # (H,W,C)
            writer.append_data(frame)
        writer.close()
        print(f"Saved tokenizer reconstruction to {args.out_video}")
        return

    if not args.var_ckpt:
        raise ValueError("Must provide --var_ckpt when not using --teacher_forced")

    # VAR model – codebook_size must match num_embeddings (256)
    cfg = VARConfig(
        codebook_size=args.num_embeddings,
        d_model=args.var_d_model,
        n_heads=args.var_n_heads,
        n_layers=args.var_n_layers,
        dropout=args.var_dropout,
    )  # VARConfig.codebook_size should be num_embeddings
    var_model = FrameConditionedVAR(cfg).to(device)
    var_model.load_state_dict(torch.load(args.var_ckpt, map_location=device))
    var_model.eval()

    with torch.no_grad():
        # encode first frame to seed tokens
        first_frame = video[:, :, 0:1, :, :]  # (1,3,1,64,64)
        tokens_list, _, _ = tokenizer.encode_tokens(first_frame)
        first_tok = tokens_list[0][:, 0]      # (1,H_tok,W_tok)
        _, H_tok, W_tok = first_tok.shape

        gen_tokens = [first_tok]

        for t in range(1, T):
            prev_tok = gen_tokens[-1]    # (1,H_tok,W_tok)
            act_prev = actions[:, t - 1] # (1,11)

            # condition on previous tokens; decode greedily for stability
            logits = var_model(prev_tok, prev_tok, act_prev)  # (1,H_tok,W_tok,V)
            logits = logits / max(args.temperature, 1e-6)
            logits = top_k_logits(logits, args.top_k)
            idx = logits.argmax(dim=-1)  # (1,H_tok,W_tok)
            gen_tokens.append(idx)

        gen_tokens_all = torch.stack(gen_tokens, dim=1)  # (1,T,H_tok,W_tok)
        recon = tokenizer.decode_tokens([gen_tokens_all])  # (1,3,T,64,64)
        recon = torch.tanh(recon)
        recon = recon.clamp(-1.0, 1.0)
        recon = (recon[0] * 0.5 + 0.5).cpu().numpy()  # (3,T,64,64)
        recon = (recon * 255.0).astype(np.uint8)

    C, T, H, W = recon.shape
    writer = imageio.get_writer(args.out_video, fps=20)
    for t in range(T):
        frame = recon[:, t].transpose(1, 2, 0)  # (H,W,C)
        writer.append_data(frame)
    writer.close()
    print(f"Saved generated video to {args.out_video}")


if __name__ == "__main__":
    main()
