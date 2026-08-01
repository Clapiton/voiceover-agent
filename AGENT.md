# @claplabs Video Voiceover Agent

An agentic pipeline that takes a silent video, analyzes it visually with GPT-4o Vision,
generates a creator-style voiceover script, synthesizes the audio with Kokoro-82M TTS,
and muxes everything back into a single MP4 using FFmpeg.

---

## Stack

| Layer | Tool | License | Cost |
|---|---|---|---|
| Frame extraction | FFmpeg | LGPL | Free |
| Visual analysis + script | GPT-4o Vision | OpenAI API | Pay-per-use (low) |
| Text-to-speech | Kokoro-82M (`hexgrad/kokoro`) | Apache 2.0 | Free |
| Audio mux | FFmpeg | LGPL | Free |

---

## Project Structure

```
claplabs-voiceover/
├── AGENT.md
├── .env
├── requirements.txt
├── voiceover.py          # main entrypoint
├── steps/
│   ├── extract_frames.py
│   ├── get_video_duration.py
│   ├── analyze_frames.py
│   ├── generate_script.py
│   ├── synthesize_voice.py
│   └── mux_audio.py
└── output/
    └── <video_name>_<timestamp>/
        ├── frames/
        ├── script.txt
        ├── voiceover.wav
        └── <video_name>_voiceover.mp4
```

---

## Environment Variables

```env
OPENAI_API_KEY=sk-...
SCRIPT_MODEL=gpt-5.4                  # model for script generation (e.g. gpt-5.4)
STYLE=energetic tech creator       # voice/tone hint passed to GPT
FRAME_RATE=0.5                     # frames per second to extract (0.5 = 1 frame every 2s)
KOKORO_VOICE=af_bella              # see voice list below
```

---

## Setup

```bash
# 1. Clone and enter project
git clone https://github.com/Clapiton/voiceover-agent.git
cd voiceover-agent

# 2. Create virtualenv
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg

# 5. Copy env file and fill in your OpenAI key
cp .env.example .env
```

### requirements.txt

```
spacy>=3.7.0,<3.8.0
openai>=1.30.0
kokoro>=0.7.0
soundfile
python-dotenv
Pillow
```

---

## Usage

```bash
python voiceover.py path/to/video.mp4
```

Optional flags:

```bash
python voiceover.py video.mp4 \
  --style "hype tech creator for Instagram Reels" \
  --fps 0.5 \
  --voice af_bella \
  --output output/final.mp4
```

---

## Pipeline Steps

### Step 1 — Frame Extraction (`steps/extract_frames.py`)

Uses FFmpeg to extract evenly spaced frames from the input video.

```python
import subprocess, os

def extract_frames(video_path: str, fps: float, output_dir: str) -> list[str]:
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
```

**Notes:**
- `fps=0.5` gives 1 frame every 2 seconds. For a 30s Reel this produces ~15 frames, well within GPT-4o's context.
- `-q:v 2` keeps JPEG quality high so GPT can read fine visual details.

---

### Step 2 — Video Duration Probe (`steps/get_video_duration.py`)

Uses `ffprobe` to read the exact duration of the video in seconds. This is the source of truth
that drives word count in the script and speed correction in the TTS step.

```python
import subprocess, json

def get_video_duration(video_path: str) -> float:
    result = subprocess.run([
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path
    ], capture_output=True, text=True, check=True)

    data = json.loads(result.stdout)

    for stream in data["streams"]:
        if stream.get("codec_type") == "video":
            return float(stream["duration"])

    raise ValueError(f"No video stream found in {video_path}")
```

**Notes:**
- `ffprobe` ships with FFmpeg, no extra install needed.
- Returns a `float` in seconds, e.g. `28.4` for a 28-second Reel.
- Falls back through all streams so it works even when stream order varies.

---

### Step 3 — Visual Analysis (`steps/analyze_frames.py`)

Encodes each frame as base64 and sends the full sequence to GPT-4o Vision.
Returns a structured scene description the script generator uses.

```python
import base64, os
from openai import OpenAI

client = OpenAI()

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_frames(frame_paths: list[str], style: str) -> str:
    image_messages = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(p)}",
                "detail": "low"   # use "high" for detailed UI/text on screen
            }
        }
        for p in frame_paths
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visual analyst for a tech creator channel called @claplabs. "
                    "You receive sequential video frames and describe what is happening "
                    "scene by scene: actions, mood, pacing, on-screen content, and visual transitions. "
                    "Be concise and structured. Format as: SCENE 1: ... SCENE 2: ..."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze these frames from a @claplabs video. Style context: {style}"},
                    *image_messages
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content
```

**Notes:**
- `detail: "low"` costs ~85 tokens per image. For 15 frames that is ~1,275 tokens on vision input, under $0.01 per video.
- Switch to `detail: "high"` if the video shows code, UI, or small text that matters for the voiceover.

---

### Step 4 — Script Generation (`steps/generate_script.py`)

Takes the scene description and the video duration, calculates a target word count,
and instructs GPT to write a script that fits exactly within that time window.

```python
from openai import OpenAI
import math

client = OpenAI()

# Natural voiceover speaking pace in words per minute.
# 130 WPM = relaxed/warm, 150 WPM = energetic/punchy creator style.
WPM = 140

PLATFORM_HINTS = {
    "reels": "Instagram Reels — hook in first 3 seconds, punchy sentences, end with a CTA.",
    "shorts": "YouTube Shorts — fast pacing, no fluff, end strong.",
    "tiktok": "TikTok — conversational, trend-aware, hook immediately, use 'you' language.",
    "linkedin": "LinkedIn video — professional but human, insight-driven, subtle CTA.",
}

def generate_script(
    scene_description: str,
    style: str,
    duration_seconds: float,
    platform: str = "reels"
) -> str:
    platform_hint = PLATFORM_HINTS.get(platform, PLATFORM_HINTS["reels"])

    # Target word count based on video duration
    target_words = math.floor((duration_seconds / 60) * WPM)

    response = client.chat.completions.create(
        model=os.getenv("SCRIPT_MODEL", "gpt-5.4"),
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are the voice of @claplabs, a Nigerian tech creator and engineer. "
                    f"Your tone is {style}. "
                    f"Write ONLY the spoken voiceover script — no stage directions, no timestamps, "
                    f"no labels. Just the words that will be spoken. "
                    f"Platform: {platform_hint} "
                    f"CRITICAL: The video is exactly {duration_seconds:.1f} seconds long. "
                    f"Your script MUST be approximately {target_words} words so it fits within that duration "
                    f"at a natural speaking pace. Do not write more or fewer words than this target."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Write the voiceover for this {duration_seconds:.1f}s video "
                    f"(target: ~{target_words} words) based on these scenes:\n\n{scene_description}"
                )
            }
        ],
        max_tokens=max(200, target_words * 2)   # tokens scale with word target
    )

    return response.choices[0].message.content.strip()
```

**Notes:**
- `WPM = 140` is a comfortable energetic pace. Lower it to `130` for a calmer delivery, raise to `150` for a fast hype style.
- `max_tokens` is dynamically set to `target_words * 2` so short videos don't get capped at an artificially low token limit.

---

### Step 5 — Voice Synthesis (`steps/synthesize_voice.py`)

Converts the script to audio using Kokoro-82M locally. After generation, the actual audio
duration is measured and compared against the target video duration. If they differ by more
than 1 second, FFmpeg `atempo` is used to time-stretch or compress the audio to fit exactly.

```python
import soundfile as sf
import numpy as np
import subprocess
import os
from kokoro import KPipeline

SAMPLE_RATE = 24000   # Kokoro fixed output sample rate

def synthesize_voice(
    script: str,
    voice: str,
    target_duration: float,
    output_path: str
) -> str:
    pipeline = KPipeline(lang_code="a")   # "a" = American English

    generator = pipeline(script, voice=voice, speed=1.0)

    samples = []
    for _, _, audio in generator:
        samples.append(audio)

    audio_combined = np.concatenate(samples)
    raw_path = output_path.replace(".wav", "_raw.wav")
    sf.write(raw_path, audio_combined, SAMPLE_RATE)

    # Measure actual generated audio duration
    actual_duration = len(audio_combined) / SAMPLE_RATE
    drift = abs(actual_duration - target_duration)

    if drift <= 1.0:
        # Close enough — just rename
        os.rename(raw_path, output_path)
        print(f"        Audio duration: {actual_duration:.2f}s (target: {target_duration:.2f}s) — no correction needed")
    else:
        # atempo range is 0.5–2.0; chain filters for extreme cases
        tempo_ratio = actual_duration / target_duration
        atempo_filter = _build_atempo_filter(tempo_ratio)

        subprocess.run([
            "ffmpeg",
            "-i", raw_path,
            "-filter:a", atempo_filter,
            output_path,
            "-y"
        ], check=True)

        os.remove(raw_path)
        print(f"        Audio duration corrected: {actual_duration:.2f}s → {target_duration:.2f}s (atempo {tempo_ratio:.3f})")

    return output_path


def _build_atempo_filter(ratio: float) -> str:
    """
    FFmpeg atempo only accepts values between 0.5 and 2.0.
    For ratios outside that range, chain multiple atempo filters.
    e.g. ratio 0.3 → atempo=0.5,atempo=0.6
    """
    filters = []
    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio /= 0.5
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0
    filters.append(f"atempo={ratio:.4f}")
    return ",".join(filters)
```

**Notes:**
- The 1-second tolerance (`drift <= 1.0`) gives a small buffer for natural pacing. Tighten to `0.5` if you need frame-perfect sync.
- `atempo` preserves pitch while changing speed, so the voice won't sound chipmunk or slowed down.
- Chaining multiple `atempo` filters handles edge cases where Kokoro generates audio wildly shorter or longer than expected.

**Available Kokoro voices (American English):**

| Voice ID | Character |
|---|---|
| `af_bella` | Warm, expressive female |
| `af_nova` | Clear, modern female |
| `af_sky` | Energetic female |
| `am_adam` | Deep, authoritative male |
| `am_michael` | Neutral, professional male |

Full voice list: `github.com/hexgrad/kokoro#voices`

---

### Step 6 — Audio Mux (`steps/mux_audio.py`)

Combines the original video (no audio) with the generated voiceover using FFmpeg.

```python
import subprocess

def mux_audio(video_path: str, audio_path: str, output_path: str) -> str:
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
```

**Notes:**
- `-c:v copy` avoids re-encoding the video, keeping quality and making it fast.
- By the time this step runs, the voiceover audio has already been time-corrected in Step 5 to match the video duration exactly, so no `-shortest` or `-t` trimming is needed.

---

### Entrypoint (`voiceover.py`)

```python
import argparse, os
from dotenv import load_dotenv
from steps.extract_frames import extract_frames
from steps.get_video_duration import get_video_duration
from steps.analyze_frames import analyze_frames
from steps.generate_script import generate_script
from steps.synthesize_voice import synthesize_voice
from steps.mux_audio import mux_audio

load_dotenv()

def run(video_path: str, style: str, fps: float, voice: str, platform: str, output_path: str):
    print("[ 1/6 ] Extracting frames...")
    frames = extract_frames(video_path, fps, output_dir="output/frames")
    print(f"        {len(frames)} frames extracted")

    print("[ 2/6 ] Probing video duration...")
    duration = get_video_duration(video_path)
    print(f"        Duration: {duration:.2f}s")

    print("[ 3/6 ] Analyzing frames with GPT-4o Vision...")
    scene_description = analyze_frames(frames, style)
    print(f"        Scene description:\n{scene_description}\n")

    print("[ 4/6 ] Generating voiceover script...")
    script = generate_script(scene_description, style, duration, platform)
    with open("output/script.txt", "w") as f:
        f.write(script)
    print(f"        Script ({len(script.split())} words):\n{script}\n")

    print("[ 5/6 ] Synthesizing and time-fitting voice with Kokoro...")
    audio_path = synthesize_voice(script, voice, target_duration=duration, output_path="output/voiceover.wav")
    print(f"        Audio saved to {audio_path}")

    print("[ 6/6 ] Muxing audio into video...")
    final = mux_audio(video_path, audio_path, output_path)
    print(f"\n Done! Final video: {final}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="@claplabs Voiceover Agent")
    parser.add_argument("video", help="Path to input video (no audio)")
    parser.add_argument("--style", default=os.getenv("STYLE", "energetic tech creator"))
    parser.add_argument("--fps", type=float, default=float(os.getenv("FRAME_RATE", 0.5)))
    parser.add_argument("--voice", default=os.getenv("KOKORO_VOICE", "af_bella"))
    parser.add_argument("--platform", default="reels", choices=["reels", "shorts", "tiktok", "linkedin"])
    parser.add_argument("--output", default="output/final_video.mp4")
    args = parser.parse_args()

    run(args.video, args.style, args.fps, args.voice, args.platform, args.output)
```

---

## Example Run

```bash
python voiceover.py demo.mp4 --style "hype Nigerian tech builder" --platform reels --voice am_adam
```

Output:

```
[ 1/6 ] Extracting frames...
        14 frames extracted
[ 2/6 ] Probing video duration...
        Duration: 28.40s
[ 3/6 ] Analyzing frames with GPT-4o Vision...
        Scene description:
        SCENE 1: Developer at laptop, terminal visible, fast typing...
        ...
[ 4/6 ] Generating voiceover script...
        Script (66 words):
        You think building an AI agent takes months? Watch this...
        ...
[ 5/6 ] Synthesizing and time-fitting voice with Kokoro...
        Audio duration corrected: 31.20s → 28.40s (atempo 1.099)
[ 6/6 ] Muxing audio into video...

 Done! Final video: output/final_video.mp4
```

---

## Estimated Cost Per Video

| Step | Tool | Est. Cost |
|---|---|---|
| Frame extraction | FFmpeg | $0 |
| Visual analysis (15 frames @ low detail) | GPT-4o Vision | ~$0.01 |
| Script generation | GPT-4o | ~$0.002 |
| TTS | Kokoro (local) | $0 |
| Mux | FFmpeg | $0 |
| **Total** | | **< $0.02** |

---

## Roadmap

- [x] Auto-detect video duration and drive word count + audio correction from it
- [ ] Add background music layer (royalty-free) via FFmpeg audio mix
- [ ] Batch mode: process a folder of videos in one command
- [ ] n8n node wrapper for Digamma workflow integration
- [ ] Voice cloning mode using F5-TTS for a consistent @claplabs voice
