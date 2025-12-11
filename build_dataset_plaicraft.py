import os
import json
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm


def load_eventlog(path):
    with open(path, "r") as f:
        events = json.load(f)
    # Ensure sorted by time
    events.sort(key=lambda e: e["unix_time"])
    return events


def load_positions(path):
    with open(path, "r") as f:
        pos_list = json.load(f)
    pos_list.sort(key=lambda p: p["timeMilli"])
    print(f"Loaded {len(pos_list)} position samples from {path}")
    return pos_list


# Map keys to indices in the one-hot part of the action vector
key_to_idx = {
    "w":  0,   # forward
    "s":  1,   # back
    "a":  2,   # left
    "d":  3,   # right
    "e":  4,   # inventory
    "space": 5,  # jump    (fixed so it does not collide with 'e')
    "Shift_L": 6,  # sneak
    "f":       7,  # swap item in hand
    "1":     8,  # hotbar slot 1
    "2":     9,  # hotbar slot 2
    "3":    10,  # hotbar slot 3
    "4":    11,  # hotbar slot 4
    "5":    12,  # hotbar slot 5
    "6":    13,  # hotbar slot 6
    "7":    14,  # hotbar slot 7
    "8":    15,  # hotbar slot 8
    "9":    16,  # hotbar slot 9
    "q":    17,  # drop item
    "Tab":  18,  # toggle player count (server only)
}



def get_nearest_position(pos_list, t_ms, last_idx=None):
    """
    Find the position sample in pos_list whose timeMilli is
    closest to t_ms (rounding in time).

    We ignore last_idx for correctness (you can keep it in the call
    signature for API compatibility), and just do a safe binary search.

    Returns:
        x, y, z, yaw, pitch, chosen_idx
    """
    n = len(pos_list)
    if n == 0:
        raise RuntimeError("pos_list is empty")

    # Clamp at ends
    t0 = pos_list[0]["timeMilli"]
    t_last = pos_list[-1]["timeMilli"]

    if t_ms <= t0:
        p = pos_list[0]
        return (
            float(p["x"]),
            float(p["y"]),
            float(p["z"]),
            float(p["yaw"]),
            float(p["pitch"]),
            0,
        )

    if t_ms >= t_last:
        p = pos_list[-1]
        return (
            float(p["x"]),
            float(p["y"]),
            float(p["z"]),
            float(p["yaw"]),
            float(p["pitch"]),
            n - 1,
        )

    # Binary search for rightmost index i with timeMilli <= t_ms
    lo, hi = 0, n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        t_mid = pos_list[mid]["timeMilli"]
        if t_mid <= t_ms:
            lo = mid + 1
        else:
            hi = mid - 1

    # hi is now the last index where timeMilli <= t_ms
    i = max(0, min(n - 1, hi))
    j = i + 1
    if j >= n:
        j = n - 1

    t_i = pos_list[i]["timeMilli"]
    t_j = pos_list[j]["timeMilli"]

    # Choose whichever is closer in time
    if abs(t_ms - t_i) <= abs(t_j - t_ms):
        idx = i
    else:
        idx = j

    p = pos_list[idx]
    return (
        float(p["x"]),
        float(p["y"]),
        float(p["z"]),
        float(p["yaw"]),
        float(p["pitch"]),
        idx,
    )


import cv2  # if you do not want cv2, I’ll show a version without it below

def build_episode_arrays(
    episode_dir,
    video_filename,
    eventlog_filename,
    position_filename,
    max_frames=None,          # NEW: cap frames per video
    target_size=(160,288),         # NEW: e.g. (360, 640) or None to keep original
):
    """
    Returns:
        frames:    (T, H, W, 3) uint8
        actions:   (T, A) float32  [keys one-hot, mouse_dx, mouse_dy]
        positions: (T, 5) float32  [x, y, z, yaw, pitch]
        key_to_idx: dict
    """
    video_path = os.path.join(episode_dir, video_filename)
    eventlog_path = os.path.join(episode_dir, eventlog_filename)
    position_path = os.path.join(episode_dir, position_filename)

    # 1) Load logs
    events = load_eventlog(eventlog_path)
    pos_list = load_positions(position_path)

    if not events:
        raise RuntimeError(f"No events found in eventlog: {eventlog_path}")
    if not pos_list:
        raise RuntimeError(f"No positions found in: {position_path}")

    # 2) Load video
    reader = imageio.get_reader(video_path)

    fps = 30.0

    print(f"[{episode_dir}] FPS: {fps}")
    dt_ms = 1000.0 / fps

    # Use get_length to avoid slow/prob buggy paths
    num_frames = reader.get_length()
    if num_frames is None or num_frames <= 0:
        raise RuntimeError(f"Could not determine num_frames for {video_path}")

    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    print(f"[{episode_dir}] num_frames (used) = {num_frames}")

    # 3) Read frames (optionally with tqdm)
    from tqdm import tqdm
    frames = []
    for i in tqdm(range(num_frames), desc=f"Loading frames [{os.path.basename(episode_dir)}]"):
        f = reader.get_data(i)  # (H, W, 3), uint8

        # Optional downscale to save memory
        if target_size is not None:
            th, tw = target_size
            f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)

        frames.append(f)

    reader.close()

    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    T = frames.shape[0]

    # 4) Prepare actions and positions
    global key_to_idx
    num_keys = len(key_to_idx)
    action_dim = num_keys + 2  # keys one hot + (mouse_dx, mouse_dy)

    actions = np.zeros((T, action_dim), dtype=np.float32)
    positions = np.zeros((T, 5), dtype=np.float32)  # x, y, z, yaw, pitch

    t0 = events[0]["unix_time"]
    frame_times = t0 + np.arange(T) * dt_ms  # Unix ms

    current_keys = set()
    event_idx = 0
    pos_idx = 0
    num_events = len(events)

    prev_yaw = None
    prev_pitch = None

    for f_idx, t_f in tqdm(enumerate(frame_times), total=len(frame_times), desc="Aligning events/positions"):
        # Position from position_data.json, rounded to nearest time sample
        x, y, z, yaw, pitch, pos_idx = get_nearest_position(pos_list, t_f, pos_idx)
        positions[f_idx] = np.array([x, y, z, yaw, pitch], dtype=np.float32)

        # Mouse deltas from yaw/pitch
        if prev_yaw is None:
            mouse_dx = 0.0
            mouse_dy = 0.0
        else:
            mouse_dx = yaw - prev_yaw
            mouse_dy = pitch - prev_pitch
        prev_yaw = yaw
        prev_pitch = pitch

        # Keys from eventlog in [t_f, t_next)
        t_next = t_f + dt_ms
        while event_idx < num_events and events[event_idx]["unix_time"] < t_next:
            ev = events[event_idx]
            etype = ev.get("event")

            if etype == "key down":
                k = ev.get("key")
                if k is not None:
                    current_keys.add(k)

            elif etype == "key up":
                k = ev.get("key")
                if k is not None and k in current_keys:
                    current_keys.remove(k)

            event_idx += 1

        # Build action vector
        a_vec = np.zeros(action_dim, dtype=np.float32)
        for k in current_keys:
            if k in key_to_idx:
                a_vec[key_to_idx[k]] = 1.0
        a_vec[-2] = mouse_dx
        a_vec[-1] = mouse_dy
        actions[f_idx] = a_vec

    return frames, actions, positions, key_to_idx





def episode_to_var_pairs(frames, actions):
    """
    Convert per-frame arrays into training pairs for a VAR-style model.

    Inputs:
        frames:    (T, H, W, 3)
        actions:   (T, A)

    Returns:
        prev_frames:     (T-1, H, W, 3)
        next_frames:     (T-1, H, W, 3)
        actions_prev:    (T-1, A)
    """
    # We want (frame_t, action_t) -> frame_{t+1}
    prev_frames = frames[:-1]
    next_frames = frames[1:]
    actions_prev = actions[:-1]
    return prev_frames, next_frames, actions_prev


def process_dataset_root(
    root_dir,
    out_dir,
    video_suffix="_video.mp4",
    eventlog_suffix="_eventlog.json",
    position_suffix="_position_data.json",
):
    """
    Walks root_dir, finds episode folders, builds arrays, and saves
    one npz per episode into out_dir.

    Assumes filenames follow pattern:
      <episode_id>_video.mp4
      <episode_id>_eventlog.json
      <episode_id>_position_data.json
    in each episode directory.

    If your structure is different, adjust this function.
    """
    os.makedirs(out_dir, exist_ok=True)
    for episode_name in os.listdir(root_dir):
        episode_dir = os.path.join(root_dir, episode_name)
        if not os.path.isdir(episode_dir):
            continue

        # Infer filenames from the directory name
        video_filename = episode_name + video_suffix
        eventlog_filename = episode_name + eventlog_suffix
        position_filename = episode_name + position_suffix

        video_path = os.path.join(episode_dir, video_filename)
        eventlog_path = os.path.join(episode_dir, eventlog_filename)
        position_path = os.path.join(episode_dir, position_filename)

        if not (os.path.exists(video_path) and
                os.path.exists(eventlog_path) and
                os.path.exists(position_path)):
            # debugging print:
            print(f"Missing files in {episode_dir}:")
            if not os.path.exists(video_path):
                print(f"  Missing video: {video_path}")
            if not os.path.exists(eventlog_path):
                print(f"  Missing eventlog: {eventlog_path}")
            if not os.path.exists(position_path):
                print(f"  Missing position data: {position_path}")
            # Skip if any of the required files is missing
            print(f"Skipping {episode_dir} - missing files")
            continue

        print(f"Processing episode: {episode_dir}")
        frames, actions, positions, _ = build_episode_arrays(
            episode_dir,
            video_filename,
            eventlog_filename,
            position_filename,
        )

        prev_frames, next_frames, actions_prev, positions_prev = episode_to_var_pairs(frames, actions, positions)

        # Save training pairs for this episode
        out_path = os.path.join(out_dir, f"{episode_name}.npz")
        np.savez_compressed(
            out_path,
            prev_frames=prev_frames,
            next_frames=next_frames,
            actions_prev=actions_prev,
            positions_prev=positions_prev,
        )
        print(f"Saved episode dataset to {out_path}")

import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root", required=True, help="Root directory containing episode folders")
#     parser.add_argument("--out", required=True, help="Output directory for npz files")
#     args = parser.parse_args()

#     process_dataset_root(args.root, args.out)


path = "/Volumes/1TBSSD/Downloads/data/c57fa94e436cf49a929d0168e47d26fec3d900b321775e280ef136979c01d5a4/00e24ef06b5b4f3b.npz"

data = np.load(path)

print("Keys stored in npz:", data.files)

actions_prev = data["actions_prev"]
print("actions_prev shape:", actions_prev.shape)

# how many rows to print
N = min(200, actions_prev.shape[0])

print(f"\nPrinting first {N} action vectors:\n")
for i in range(N):
    print(f"{i}: {actions_prev[i]}")

print(sum(actions_prev))