import base64
import os
from typing import List
from openai import OpenAI

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_frames(frame_paths: List[str], style: str, model: str = "gpt-4o") -> str:
    """
    Sends frame sequence to OpenAI Vision model for visual scene analysis.
    """
    client = OpenAI()

    image_messages = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(p)}",
                "detail": "low"
            }
        }
        for p in frame_paths
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
        max_tokens=1000
    )

    return response.choices[0].message.content
