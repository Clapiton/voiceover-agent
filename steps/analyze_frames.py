import base64
import os
from typing import List
from openai import OpenAI

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_frames(frame_paths: List[str], style: str, model: str = "gpt-4o", max_frames: int = 16) -> str:
    """
    Sends frame sequence to OpenAI Vision model for visual scene analysis.
    Subsamples frames if count exceeds max_frames to keep request size lightweight and reliable.
    """
    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        selected_frames = [frame_paths[i] for i in indices]
    else:
        selected_frames = frame_paths

    client = OpenAI(timeout=60.0, max_retries=3)

    image_messages = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(p)}",
                "detail": "low"
            }
        }
        for p in selected_frames
    ]

    response = client.chat.completions.create(
        model=model,
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
        max_completion_tokens=1000
    )

    return response.choices[0].message.content
