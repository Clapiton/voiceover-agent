import os
import subprocess
import soundfile as sf
import numpy as np
from kokoro import KPipeline

SAMPLE_RATE = 24000   # Kokoro fixed output sample rate

def _build_atempo_filter(ratio: float) -> str:
    """
    FFmpeg atempo only accepts values between 0.5 and 2.0.
    For ratios outside that range, chain multiple atempo filters.
    e.g. ratio 0.3 -> atempo=0.5,atempo=0.6
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

def synthesize_voice(
    script: str,
    voice: str,
    target_duration: float,
    output_path: str
) -> str:
    """
    Converts script text into spoken audio with Kokoro-82M TTS and adjusts speed via FFmpeg atempo if needed.
    """
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
        # Close enough — rename raw output
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(raw_path, output_path)
        print(f"        Audio duration: {actual_duration:.2f}s (target: {target_duration:.2f}s) — no correction needed")
    else:
        # Tempo ratio calculation: audio too long -> ratio > 1 (speed up); audio too short -> ratio < 1 (slow down)
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
        print(f"        Audio duration corrected: {actual_duration:.2f}s -> {target_duration:.2f}s (atempo {tempo_ratio:.3f})")

    return output_path
