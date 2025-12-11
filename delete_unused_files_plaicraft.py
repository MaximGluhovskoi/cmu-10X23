#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import argparse

TARGET_DIR_NAMES = {"encoded_audio_continuous", "encoded_video"}
TARGET_FILE_SUFFIXES = {
    ".db",
    "_audio_in.wav",
    "_audio_out.wav",
}

def should_delete_file(name: str) -> bool:
    if name.endswith(".db"):
        return True
    if name.endswith("_audio_in.wav"):
        return True
    if name.endswith("_audio_out.wav"):
        return True
    return False

def collect_paths(root: Path):
    dirs_to_delete = []
    files_to_delete = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        # collect directories
        for d in dirnames:
            if d in TARGET_DIR_NAMES:
                full = dirpath / d
                dirs_to_delete.append(full)

        # collect files
        for f in filenames:
            if should_delete_file(f):
                full = dirpath / f
                files_to_delete.append(full)

    return dirs_to_delete, files_to_delete

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root of Plaicraft data (e.g., /Volumes/1TBSSD/Downloads/data)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root path does not exist: {root}")
        return

    print(f"Scanning under: {root}")
    dirs_to_delete, files_to_delete = collect_paths(root)

    print(f"\nFound {len(dirs_to_delete)} directories to delete:")
    for d in dirs_to_delete[:10]:
        print(f"  [dir] {d}")
    if len(dirs_to_delete) > 10:
        print(f"  ... and {len(dirs_to_delete) - 10} more")

    print(f"\nFound {len(files_to_delete)} files to delete:")
    for f in files_to_delete[:10]:
        print(f"  [file] {f}")
    if len(files_to_delete) > 10:
        print(f"  ... and {len(files_to_delete) - 10} more")

    confirm = input("\nProceed with deletion? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborting, nothing deleted.")
        return

    # delete files first
    print("\nDeleting files...")
    for f in files_to_delete:
        if f.exists():
            try:
                print(f"  Deleting file: {f}")
                f.unlink()
            except OSError as e:
                print(f"  Warning: could not delete file {f}: {e}")
        else:
            # debug: file missing already
            # print(f"  Skipping missing file: {f}")
            pass

    # delete directories deepest-first so children go first
    import subprocess
    import errno

    print("\nDeleting directories...")
    dirs_to_delete_sorted = sorted(dirs_to_delete, key=lambda p: len(str(p)), reverse=True)

    for d in dirs_to_delete_sorted:
        if not d.exists():
            # Already gone, nothing to do
            continue

        try:
            print(f"  Deleting dir  : {d}")
            shutil.rmtree(d)
        except OSError as e:
            print(f"  Warning: shutil.rmtree failed for {d}: {e}")

            # Extra debug
            parent = d.parent
            if parent.exists():
                print(f"    Parent dir contents of {parent}:")
                try:
                    for name in os.listdir(parent):
                        print(f"      {name!r}")
                except Exception as e2:
                    print(f"    Could not list parent: {e2}")

            # If it looks like ENOENT weirdness, fall back to rm -rf
            try:
                print(f"    Falling back to: rm -rf '{d}'")
                subprocess.run(
                    ["rm", "-rf", str(d)],
                    check=False,
                )
                # Check after rm -rf
                if d.exists():
                    print(f"    After rm -rf: still exists: {d}")
                else:
                    print(f"    After rm -rf: removed {d}")
            except Exception as e3:
                print(f"    Fallback rm -rf failed for {d}: {e3}")


    print("\nDone.")

if __name__ == "__main__":
    main()
