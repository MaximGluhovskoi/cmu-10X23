import argparse
import os
import torch

# This matches open-oasis utils.py exactly
ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

CAMERA_MAX_VAL = 20.0
CAMERA_BIN_SIZE = 0.5
CAMERA_NUM_BUCKETS = int(CAMERA_MAX_VAL / CAMERA_BIN_SIZE)  # 40


def load_actions_tensor(path: str) -> torch.Tensor:
    """
    Load the actions tensor as saved in *.one_hot_actions.pt.
    In open-oasis, this should have shape [T, len(ACTION_KEYS)].
    """
    actions = torch.load(path)
    if not torch.is_tensor(actions):
        raise TypeError(f"Expected tensor, got {type(actions)}")
    if actions.dim() != 2 or actions.shape[1] != len(ACTION_KEYS):
        raise ValueError(
            f"Expected shape [T, {len(ACTION_KEYS)}], got {tuple(actions.shape)}"
        )
    return actions


def decode_camera_value(norm_val: float):
    """
    In one_hot_actions, camera bucket index 'b' is mapped to:
        norm_val = (b - CAMERA_NUM_BUCKETS) / CAMERA_NUM_BUCKETS
    where b is in [0, 2 * CAMERA_NUM_BUCKETS] (0..80).

    Here we approximately invert that:
        b = round(norm_val * CAMERA_NUM_BUCKETS + CAMERA_NUM_BUCKETS)
        degrees = b * CAMERA_BIN_SIZE - CAMERA_MAX_VAL  (range about [-20, 20])
    """
    b = int(round(norm_val * CAMERA_NUM_BUCKETS + CAMERA_NUM_BUCKETS))
    # clamp to valid bucket range just in case
    b = max(0, min(2 * CAMERA_NUM_BUCKETS, b))
    degrees = b * CAMERA_BIN_SIZE - CAMERA_MAX_VAL
    return b, degrees


def pretty_print_actions(actions: torch.Tensor, max_timesteps: int | None = None):
    """
    Print actions in a human readable format.

    For each timestep t:
      - show discrete actions that are active (value > 0.5)
      - decode cameraX and cameraY back to bucket and degrees
    """
    T, D = actions.shape
    if max_timesteps is not None:
        T = min(T, max_timesteps)

    print(f"Actions tensor shape: {actions.shape}")
    print(f"Printing first {T} timesteps\n")

    for t in range(T):
        row = actions[t]
        print(f"t = {t}")

        # Discrete keys (everything except cameraX and cameraY)
        active = []
        for idx, key in enumerate(ACTION_KEYS):
            if key.startswith("camera"):
                continue
            val = float(row[idx])
            if val > 0.5:
                active.append(key)

        if active:
            print("  active discrete actions:", ", ".join(active))
        else:
            print("  active discrete actions: (none)")

        # Camera values
        cam_x = float(row[ACTION_KEYS.index("cameraX")])
        cam_y = float(row[ACTION_KEYS.index("cameraY")])

        # Only print camera info if there is any noticeable movement
        eps = 1e-4
        if abs(cam_x) > eps or abs(cam_y) > eps:
            bx, deg_x = decode_camera_value(cam_x)
            by, deg_y = decode_camera_value(cam_y)
            print(
                f"  cameraX: norm={cam_x:+.3f}, bucket={bx:2d}, approx_deg={deg_x:+.2f}"
            )
            print(
                f"  cameraY: norm={cam_y:+.3f}, bucket={by:2d}, approx_deg={deg_y:+.2f}"
            )
        else:
            print("  cameraX/cameraY: no movement")

        print()  # blank line between timesteps


def main():
    parser = argparse.ArgumentParser(
        description="Decode open-oasis *.one_hot_actions.pt into readable actions."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to *.one_hot_actions.pt",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=50,
        help="Max timesteps to print (for long trajectories). Use -1 for all.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"File not found: {args.path}")

    actions = load_actions_tensor(args.path)

    max_t = None if args.max_timesteps == -1 else args.max_timesteps
    pretty_print_actions(actions, max_timesteps=max_t)


if __name__ == "__main__":
    main()
