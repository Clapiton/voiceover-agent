import subprocess

def mux_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Combines original video stream with synthesized voice audio into final MP4.
    """
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",       # video stream from original
        "-map", "1:a",       # audio stream from voiceover
        "-c:v", "copy",      # no re-encode on video
        "-c:a", "aac",       # encode audio to AAC for MP4 compatibility
        "-shortest",         # trim to shortest stream
        output_path,
        "-y"
    ], check=True)

    return output_path
