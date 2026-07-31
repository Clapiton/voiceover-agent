import math
import os
from openai import OpenAI

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
    platform: str = "reels",
    model: str = None
) -> str:
    """
    Generates a spoken voiceover script fitted to the exact video duration
    using the latest GPT model (default: gpt-4.5-preview or configured SCRIPT_MODEL).
    """
    if not model:
        model = os.getenv("SCRIPT_MODEL", "gpt-4.5-preview")

    client = OpenAI()
    platform_hint = PLATFORM_HINTS.get(platform.lower(), PLATFORM_HINTS["reels"])

    # Target word count based on video duration
    target_words = math.floor((duration_seconds / 60) * WPM)

    response = client.chat.completions.create(
        model=model,
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
        max_tokens=max(200, target_words * 2)
    )

    return response.choices[0].message.content.strip()
