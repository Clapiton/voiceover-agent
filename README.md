# @claplabs Video Voiceover Agent

An agentic pipeline that takes a silent video, analyzes it visually with **GPT-4o Vision**, generates a creator-style voiceover script using the latest **GPT model** (default: `gpt-4.5-preview`), synthesizes the audio with **Kokoro-82M TTS**, and muxes everything back into a single MP4 using **FFmpeg**.

GitHub Repository: [https://github.com/Clapiton/voiceover-agent.git](https://github.com/Clapiton/voiceover-agent.git)

---

## Tech Stack

| Layer | Tool | License | Cost |
|---|---|---|---|
| Frame extraction | FFmpeg | LGPL | Free |
| Visual analysis | GPT-4o Vision | OpenAI API | Pay-per-use |
| Script generation | GPT (Latest: `gpt-4.5-preview`) | OpenAI API | Pay-per-use |
| Text-to-speech | Kokoro-82M (`hexgrad/kokoro`) | Apache 2.0 | Free |
| Audio mux | FFmpeg | LGPL | Free |

---

## Project Structure

```
voiceover-agent/
├── AGENT.md
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── voiceover.py          # main entrypoint
├── steps/
│   ├── __init__.py
│   ├── extract_frames.py
│   ├── get_video_duration.py
│   ├── analyze_frames.py
│   ├── generate_script.py
│   ├── synthesize_voice.py
│   └── mux_audio.py
└── output/
    ├── frames/
    ├── script.txt
    ├── voiceover.wav
    └── final_video.mp4
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Clapiton/voiceover-agent.git
cd voiceover-agent
```

### 2. Environment Setup

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
SCRIPT_MODEL=gpt-4.5-preview
STYLE=energetic tech creator
FRAME_RATE=0.5
KOKORO_VOICE=af_bella
```

---

## Usage

Run the agent on any silent MP4 video:

```bash
python voiceover.py path/to/video.mp4
```

### Optional Command-Line Options

```bash
python voiceover.py video.mp4 \
  --style "hype tech creator for Instagram Reels" \
  --script-model "gpt-4.5-preview" \
  --fps 0.5 \
  --voice af_bella \
  --platform reels \
  --output output/final.mp4
```

Available `--platform` options: `reels`, `shorts`, `tiktok`, `linkedin`.

---

## Pipeline Overview

1. **Frame Extraction**: FFmpeg extracts frames at `--fps` interval.
2. **Duration Probe**: `ffprobe` determines exact video duration.
3. **Visual Analysis**: GPT-4o Vision analyzes frame sequence to describe scenes.
4. **Script Generation**: Latest GPT model (`gpt-4.5-preview`) writes voiceover script tailored to target word count based on duration and platform style.
5. **Voice Synthesis**: Kokoro-82M generates audio; FFmpeg `atempo` aligns audio duration with video.
6. **Muxing**: FFmpeg joins audio and original video without re-encoding video.

---

## License

MIT License
