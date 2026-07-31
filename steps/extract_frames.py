import subprocess
import os
from typing import List

def extract_frames(video_path: str, fps: float, output_dir: str) -> List[str]:
    """
    Extracts frames from video using FFmpeg at specified frame rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern,
        "-y"
    ], check=True)

    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jpg")
    ])

    return frames
