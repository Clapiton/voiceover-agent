from datetime import datetime
import argparse
import os
from dotenv import load_dotenv

from steps.extract_frames import extract_frames
from steps.get_video_duration import get_video_duration
from steps.analyze_frames import analyze_frames
from steps.generate_script import generate_script
from steps.synthesize_voice import synthesize_voice
from steps.mux_audio import mux_audio

load_dotenv()

def get_unique_path(path: str) -> str:
    """If target path exists, appends _1, _2, etc. before extension to avoid overwriting."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def run(video_path: str, style: str, fps: float, voice: str, platform: str, output_path: str = None, script_model: str = "gpt-5.4"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Dedicated timestamped run directory to store all artifacts for this run
    run_dir = os.path.join("output", f"{video_basename}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    frames_dir = os.path.join(run_dir, "frames")
    script_file = os.path.join(run_dir, "script.txt")
    audio_file = os.path.join(run_dir, "voiceover.wav")

    if not output_path or output_path == "output/final_video.mp4":
        final_output = os.path.join(run_dir, f"{video_basename}_voiceover.mp4")
    else:
        final_output = get_unique_path(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(final_output)), exist_ok=True)

    print(f"[ 1/6 ] Extracting frames to {frames_dir}...")
    frames = extract_frames(video_path, fps, output_dir=frames_dir)
    print(f"        {len(frames)} frames extracted")

    print("[ 2/6 ] Probing video duration...")
    duration = get_video_duration(video_path)
    print(f"        Duration: {duration:.2f}s")

    print("[ 3/6 ] Analyzing frames with GPT-4o Vision...")
    scene_description = analyze_frames(frames, style)
    print(f"        Scene description:\n{scene_description}\n")

    print(f"[ 4/6 ] Generating voiceover script using model '{script_model}'...")
    script = generate_script(scene_description, style, duration, platform, model=script_model)
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"        Script ({len(script.split())} words) saved to {script_file}:\n{script}\n")

    print("[ 5/6 ] Synthesizing and time-fitting voice with Kokoro...")
    audio_path = synthesize_voice(script, voice, target_duration=duration, output_path=audio_file)
    print(f"        Audio saved to {audio_path}")

    print("[ 6/6 ] Muxing audio into video...")
    final = mux_audio(video_path, audio_path, final_output)
    print(f"\n Done! Final video saved to: {final}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="@claplabs Voiceover Agent")
    parser.add_argument("video", help="Path to input video (no audio)")
    parser.add_argument("--style", default=os.getenv("STYLE", "energetic tech creator"))
    parser.add_argument("--fps", type=float, default=float(os.getenv("FRAME_RATE", 0.5)))
    parser.add_argument("--voice", default=os.getenv("KOKORO_VOICE", "af_bella"))
    parser.add_argument("--platform", default="reels", choices=["reels", "shorts", "tiktok", "linkedin"])
    parser.add_argument("--script-model", default=os.getenv("SCRIPT_MODEL", "gpt-5.4"), help="GPT model for script generation")
    parser.add_argument("--output", default=None, help="Custom output video path")
    args = parser.parse_args()

    run(args.video, args.style, args.fps, args.voice, args.platform, args.output, args.script_model)
