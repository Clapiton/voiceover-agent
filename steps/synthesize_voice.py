import os
import gc
import subprocess
import soundfile as sf
import numpy as np

# Limit PyTorch CPU thread memory overhead on constrained cloud environments
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

SAMPLE_RATE = 24000   # Kokoro fixed output sample rate

def _build_atempo_filter(ratio: float) -> str:
    filters = []
    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio /= 0.5
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0
    filters.append(f"atempo={ratio:.4f}")
    return ",".join(filters)

def _synthesize_openai_tts(script: str, voice: str, output_path: str):
    """Zero-RAM Cloud TTS fallback using OpenAI tts-1 model."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Map Kokoro voices to closest OpenAI voices
    voice_map = {
        "af_bella": "nova",
        "af_nicole": "shimmer",
        "am_adam": "onyx",
        "am_michael": "echo",
        "bf_emma": "fable",
        "bm_george": "alloy"
    }
    tts_voice = voice_map.get(voice, "alloy")
    
    response = client.audio.speech.create(
        model="tts-1",
        voice=tts_voice,
        input=script
    )
    
    response.stream_to_file(output_path)
    return output_path

def synthesize_voice(
    script: str,
    voice: str,
    target_duration: float,
    output_path: str
) -> str:
    """
    Converts script text into spoken audio. Uses Kokoro-82M with strict memory management, 
    and automatically falls back to lightweight OpenAI Cloud TTS if RAM is constrained.
    """
    # Force garbage collection before running
    gc.collect()

    try:
        from kokoro import KPipeline
        print("        Synthesizing with Kokoro-82M TTS (memory optimized)...")
        pipeline = KPipeline(lang_code="a")
        
        with torch.no_grad():
            generator = pipeline(script, voice=voice, speed=1.0)
            samples = []
            for _, _, audio in generator:
                samples.append(audio)

        audio_combined = np.concatenate(samples)
        raw_path = output_path.replace(".wav", "_raw.wav")
        sf.write(raw_path, audio_combined, SAMPLE_RATE)

        actual_duration = len(audio_combined) / SAMPLE_RATE
        drift = abs(actual_duration - target_duration)

        if drift <= 1.0:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(raw_path, output_path)
        else:
            tempo_ratio = actual_duration / target_duration
            atempo_filter = _build_atempo_filter(tempo_ratio)

            subprocess.run([
                "ffmpeg",
                "-i", raw_path,
                "-filter:a", atempo_filter,
                output_path,
                "-y"
            ], check=True)

            if os.path.exists(raw_path):
                os.remove(raw_path)

        # Cleanup memory immediately
        del pipeline, samples, audio_combined
        gc.collect()
        return output_path

    except Exception as e:
        print(f"        Kokoro RAM memory limit hit or error ({str(e)}). Switching to zero-RAM OpenAI Cloud TTS fallback...")
        gc.collect()
        return _synthesize_openai_tts(script, voice, output_path)
