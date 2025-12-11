# count_frames.py
import sys
import imageio.v2 as imageio

mp4 = sys.argv[1]
reader = imageio.get_reader(mp4)
n = reader.get_length()
reader.close()

print(f"{mp4} has {n} frames.")
