import os
import sys
import uuid
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path to re-use existing pipeline steps without altering root project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.extract_frames import extract_frames
from steps.get_video_duration import get_video_duration
from steps.analyze_frames import analyze_frames
from steps.generate_script import generate_script
from steps.synthesize_voice import synthesize_voice
from steps.mux_audio import mux_audio

app = FastAPI(
    title="@claplabs Voiceover Agent API",
    description="Backend REST & Background Task API for Mobile (iOS/Android) Video Voiceover Agent",
    version="1.0.0"
)

# Enable CORS for cross-platform mobile apps & web preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task tracker (Production could use Redis/Celery)
TASKS_DB: Dict[str, Dict[str, Any]] = {}

# Ensure output and temp uploads directories exist
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static output files
app.mount("/static/output", StaticFiles(directory=OUTPUT_DIR), name="output")


class VoiceoverConfig(BaseModel):
    style: str = "energetic tech creator"
    fps: float = 0.5
    voice: str = "af_bella"
    platform: str = "reels"
    script_model: str = "gpt-5.4"


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "service": "Voiceover Agent Backend"}


@app.get("/api/v1/options")
def get_options():
    """Returns available Kokoro voices, platform presets, and script models."""
    return {
        "voices": [
            {"id": "af_bella", "name": "Bella", "gender": "Female", "description": "Energetic, clear, creator-focused"},
            {"id": "af_nicole", "name": "Nicole", "gender": "Female", "description": "Calm, educational, smooth"},
            {"id": "am_adam", "name": "Adam", "gender": "Male", "description": "Deep, authoritative, tech review"},
            {"id": "am_michael", "name": "Michael", "gender": "Male", "description": "Dynamic, storytelling, hype"},
            {"id": "bf_emma", "name": "Emma (UK)", "gender": "Female", "description": "British accent, sophisticated"},
            {"id": "bm_george", "name": "George (UK)", "gender": "Male", "description": "British accent, narrator"}
        ],
        "platforms": [
            {"id": "reels", "name": "Instagram Reels", "max_length": 60, "style": "Fast-paced, hook in first 3s"},
            {"id": "tiktok", "name": "TikTok", "max_length": 60, "style": "Casual, trending tone, energetic"},
            {"id": "shorts", "name": "YouTube Shorts", "max_length": 60, "style": "Punchy, concise, high engagement"},
            {"id": "linkedin", "name": "LinkedIn Video", "max_length": 120, "style": "Professional, insightful, value-first"}
        ],
        "style_presets": [
            "energetic tech creator",
            "cinematic hype storyteller",
            "calm educational tutorial",
            "ASMR gadget reviewer",
            "funny observational commentary"
        ],
        "script_models": ["gpt-5.4", "gpt-4o", "gpt-4o-mini"]
    }


def run_pipeline_task(
    task_id: str,
    video_path: str,
    style: str,
    fps: float,
    voice: str,
    platform: str,
    script_model: str
):
    try:
        task = TASKS_DB[task_id]
        task["status"] = "processing"
        
        # Step 1: Extract frames
        task["step"] = 1
        task["step_label"] = "Extracting video frames"
        task["progress"] = 15
        task["logs"].append(f"[1/6] Extracting frames at {fps} FPS...")
        
        task_run_dir = os.path.join(OUTPUT_DIR, task_id)
        frames_dir = os.path.join(task_run_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        frames = extract_frames(video_path, fps, output_dir=frames_dir)
        task["logs"].append(f"Successfully extracted {len(frames)} frames.")
        
        # Step 2: Probing video duration
        task["step"] = 2
        task["step_label"] = "Probing duration"
        task["progress"] = 30
        duration = get_video_duration(video_path)
        task["duration"] = duration
        task["logs"].append(f"[2/6] Probed duration: {duration:.2f}s")
        
        # Step 3: Visual analysis
        task["step"] = 3
        task["step_label"] = "Analyzing scenes with GPT-4o Vision"
        task["progress"] = 50
        task["logs"].append("[3/6] Analyzing frames visually...")
        scene_description = analyze_frames(frames, style)
        task["scene_description"] = scene_description
        task["logs"].append("Scene analysis complete.")
        
        # Step 4: Generate voiceover script
        task["step"] = 4
        task["step_label"] = f"Generating script ({script_model})"
        task["progress"] = 70
        task["logs"].append(f"[4/6] Drafting script using model '{script_model}'...")
        script = generate_script(scene_description, style, duration, platform, model=script_model)
        task["script"] = script
        task["word_count"] = len(script.split())
        task["logs"].append(f"Generated {task['word_count']}-word script.")
        
        # Step 5: Synthesize voice
        task["step"] = 5
        task["step_label"] = "Synthesizing TTS Audio with Kokoro"
        task["progress"] = 85
        audio_file = os.path.join(task_run_dir, "voiceover.wav")
        task["logs"].append(f"[5/6] Synthesizing voice with voice '{voice}'...")
        audio_path = synthesize_voice(script, voice, target_duration=duration, output_path=audio_file)
        task["logs"].append("Audio synthesis & time-fitting complete.")
        
        # Step 6: Muxing audio into video
        task["step"] = 6
        task["step_label"] = "Muxing final video & audio"
        task["progress"] = 95
        output_file = os.path.join(task_run_dir, "final_voiceover.mp4")
        task["logs"].append("[6/6] Muxing audio into output video...")
        final_path = mux_audio(video_path, audio_path, output_file)
        
        task["status"] = "completed"
        task["progress"] = 100
        task["step_label"] = "Completed"
        task["output_video_url"] = f"/static/output/{task_id}/final_voiceover.mp4"
        task["logs"].append(f"Done! Output saved to: {final_path}")
        
    except Exception as e:
        TASKS_DB[task_id]["status"] = "failed"
        TASKS_DB[task_id]["error"] = str(e)
        TASKS_DB[task_id]["logs"].append(f"ERROR: {str(e)}")


@app.post("/api/v1/voiceover/process")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    style: str = Form("energetic tech creator"),
    fps: float = Form(0.5),
    voice: str = Form("af_bella"),
    platform: str = Form("reels"),
    script_model: str = Form("gpt-5.4")
):
    """Submits a new video for voiceover processing."""
    task_id = str(uuid.uuid4())[:8]
    input_video_path = os.path.join(UPLOAD_DIR, f"{task_id}_{video.filename}")
    
    with open(input_video_path, "wb") as f:
        content = await video.read()
        f.write(content)
        
    TASKS_DB[task_id] = {
        "task_id": task_id,
        "filename": video.filename,
        "status": "queued",
        "progress": 0,
        "step": 0,
        "step_label": "Queued",
        "duration": 0.0,
        "scene_description": "",
        "script": "",
        "word_count": 0,
        "config": {
            "style": style,
            "fps": fps,
            "voice": voice,
            "platform": platform,
            "script_model": script_model
        },
        "logs": [f"Task {task_id} initialized for video '{video.filename}'."],
        "output_video_url": None,
        "error": None
    }
    
    background_tasks.add_task(
        run_pipeline_task,
        task_id,
        input_video_path,
        style,
        fps,
        voice,
        platform,
        script_model
    )
    
    return JSONResponse(status_code=202, content={
        "task_id": task_id,
        "message": "Video uploaded successfully. Task started.",
        "status_url": f"/api/v1/voiceover/status/{task_id}"
    })


@app.get("/api/v1/voiceover/status/{task_id}")
def get_task_status(task_id: str):
    """Polling endpoint for mobile app to retrieve real-time pipeline status."""
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS_DB[task_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
