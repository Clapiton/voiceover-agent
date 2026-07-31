import subprocess
import json

def get_video_duration(video_path: str) -> float:
    """
    Probes video duration in seconds using ffprobe.
    """
    result = subprocess.run([
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path
    ], capture_output=True, text=True, check=True)

    data = json.loads(result.stdout)

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream["duration"])

    raise ValueError(f"No video stream found in {video_path}")
