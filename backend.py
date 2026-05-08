#!/usr/bin/env python3
"""
Curiosity Trap – Video Generation Backend
Provides endpoints for script generation, TTS, image fetching, video composition, and subtitles.
"""

import os
import json
import io
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import edge_tts  # For realistic, natural-sounding TTS
from moviepy import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
    ColorClip,
    vfx,
)
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2

# ---------- Configuration ----------
PEXELS_API_URL = "https://api.pexels.com/v1/search"
GEMINI_MODEL = "gemini-1.5-flash"  # or "gemini-2.0-flash-exp"

# ---------- FastAPI App ----------
app = FastAPI(title="Curiosity Trap Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helper: Extract keywords for image search ----------
def extract_keywords(script_data: dict) -> List[str]:
    keywords = set()
    for ch in script_data.get("chapters", []):
        for w in ch.get("title", "").split():
            if len(w) > 3:
                keywords.add(w.lower())
    for w in script_data.get("title", "").split():
        if len(w) > 3:
            keywords.add(w.lower())
    for ent in script_data.get("keyEntities", []):
        keywords.add(ent.lower())
    return list(keywords)[:20]

# ---------- Endpoint: Script Generation ----------
@app.post("/api/generate-script")
async def generate_script(data: dict):
    theme = data.get("theme")
    api_key = data.get("apiKey")
    if not theme or not api_key:
        raise HTTPException(400, "theme and apiKey required")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are a script writer for the documentary series "The Curiosity Trap". Write a script for an ~8 minute video on the theme: "{theme}".

Structure:
- Title: bold and provocative.
- Hook: a gap in knowledge, a void effect.
- 6–8 short chapters, each with a suggestive one-line title.
- For each chapter, 3–5 sentences of narration (under 12 words per sentence).
- Use dramatic adjectives, personification, contrasts.
- End with an open reflection or warning.
- Include a list of key entities (people, places, things).

Output a JSON object with these keys:
{{
  "title": "string",
  "hook": "string",
  "chapters": [
    {{
      "title": "string",
      "text": "string (narration for that chapter)",
      "keyword": "single most important visual keyword"
    }}
  ],
  "keyEntities": ["Entity1", "Entity2"],
  "fullText": "the entire narration as a single string, with chapter titles as separators like 'Chapter: TITLE'"
}}
Return only valid JSON, no extra text.
"""

    response = model.generate_content(prompt)
    try:
        # Parse JSON from response text (strip potential markdown fences)
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        script = json.loads(raw)
        # Ensure fullText exists
        if "fullText" not in script:
            # Build fullText from chapters
            full = ""
            for ch in script["chapters"]:
                full += f"Chapter: {ch['title']}. {ch['text']} "
            script["fullText"] = full.strip()
        return script
    except Exception as e:
        raise HTTPException(500, f"Failed to parse Gemini output: {e}")

# ---------- Endpoint: TTS Generation ----------
@app.post("/api/generate-tts")
async def generate_tts(data: dict):
    text = data.get("text")
    voice = data.get("voice", "en-US-ChristopherNeural")  # male, deep
    if not text:
        raise HTTPException(400, "text required")

    # Use edge-tts (free, high quality)
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        await communicate.save(tmp.name)
        tmp_path = tmp.name

    # Read into BytesIO for streaming
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")

# ---------- Endpoint: Fetch Images ----------
@app.post("/api/fetch-images")
async def fetch_images(data: dict):
    keywords = data.get("keywords", [])
    count = data.get("count", 15)
    api_key = data.get("apiKey")
    if not api_key:
        raise HTTPException(400, "apiKey required")

    headers = {"Authorization": api_key}
    all_urls = []
    per_page = min(count, 10)  # Pexels max per page

    for kw in keywords[:5]:  # limit to top 5 keywords
        params = {"query": kw, "per_page": per_page, "orientation": "landscape"}
        resp = requests.get(PEXELS_API_URL, headers=headers, params=params)
        if resp.status_code == 200:
            data_resp = resp.json()
            for photo in data_resp.get("photos", []):
                all_urls.append(photo["src"]["large"])
                if len(all_urls) >= count:
                    break
        if len(all_urls) >= count:
            break

    if not all_urls:
        raise HTTPException(500, "No images found. Try different keywords.")

    return {"urls": all_urls[:count]}

# ---------- Video Composition Helpers ----------
def download_image(url, max_size=2000):
    """Download image and resize to max dimension."""
    resp = requests.get(url, stream=True)
    resp.raw.decode_content = True
    img = Image.open(resp.raw).convert("RGB")
    # Resize to manageable size while keeping aspect ratio
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return np.array(img)

def apply_vignette(clip, intensity=0.6):
    """Add dark vignette effect."""
    def vignette_effect(get_frame, t):
        frame = get_frame(t).copy()
        h, w = frame.shape[:2]
        # Create elliptical mask
        X, Y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
        mask = 1 - np.sqrt(X**2 + Y**2) * 0.9
        mask = np.clip(mask, 0, 1)
        mask = mask ** 1.5
        frame = (frame * mask[..., np.newaxis]).astype(np.uint8)
        return frame
    return clip.fl(vignette_effect)

def ken_burns_zoom(clip, zoom_ratio=1.03, duration=None):
    """Apply slow zoom in (Ken Burns) with slight randomness."""
    if duration is None:
        duration = clip.duration
    # Zoom from 1.0 to zoom_ratio
    def make_frame(t):
        factor = 1.0 + (zoom_ratio - 1.0) * (t / duration) if duration>0 else 1.0
        # We'll resize the image accordingly
        # This is better done with moviepy's resize effect, but for simplicity:
        return clip.get_frame(t)  # placeholder
    # Use moviepy's `resize` with time-dependent lambda
    return clip.resized(lambda t: 1.0 + (zoom_ratio-1.0)*(t/duration) if duration else 1.0)

def build_video(script_data, audio_bytes, image_urls):
    """Compose the final video clip."""
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tmp_audio.write(audio_bytes)
        audio_path = tmp_audio.name
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration

    # Download images
    images = []
    for url in image_urls:
        try:
            img = download_image(url)
            images.append(img)
        except:
            continue
    if not images:
        raise ValueError("No valid images downloaded")

    # Build image clips with effects
    image_clips = []
    chapters = script_data.get("chapters", [])
    # Determine average time per image: spread across total duration, but give each ~8 sec
    num_images = len(images)
    avg_dur_per_image = max(4.0, total_duration / num_images)

    for i, img in enumerate(images):
        dur = avg_dur_per_image
        # Create clip
        clip = ImageClip(img).set_duration(dur)
        # Resize to 1920x1080
        clip = clip.resized(newsize=(1920, 1080))
        # Ken Burns zoom
        clip = ken_burns_zoom(clip, zoom_ratio=1.05, duration=dur)
        # Vignette
        clip = apply_vignette(clip, intensity=0.6)
        # Color grading: slight desaturation and contrast
        clip = clip.fx(vfx.colorx, 0.85)  # reduce brightness slightly
        image_clips.append(clip)

    # Concatenate image clips (they will loop if not enough)
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    video_without_audio = concatenate_videoclips(image_clips, method="compose")
    # Trim to audio length
    if video_without_audio.duration > total_duration:
        video_without_audio = video_without_audio.subclip(0, total_duration)
    else:
        # Loop if too short
        n_loops = int(np.ceil(total_duration / video_without_audio.duration))
        clips_loop = [video_without_audio] * n_loops
        video_without_audio = concatenate_videoclips(clips_loop).subclip(0, total_duration)

    # Set audio
    final_video = video_without_audio.set_audio(audio_clip)

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        final_video.write_videofile(
            tmp_video.name,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="ultrafast",
            logger=None,
            verbose=False,
        )
        video_path = tmp_video.name

    # Cleanup audio temp
    os.unlink(audio_path)
    return video_path

# ---------- Endpoint: Video Composition ----------
@app.post("/api/compose-video")
async def compose_video(
    script: str = Form(...),
    audio: UploadFile = File(...),
    imageUrls: str = Form(...),
    duration: str = Form("480"),
):
    script_data = json.loads(script)
    image_urls = json.loads(imageUrls)
    audio_bytes = await audio.read()

    # Run video building in a thread pool because moviepy is CPU intensive
    loop = asyncio.get_running_loop()
    video_path = await loop.run_in_executor(
        None, build_video, script_data, audio_bytes, image_urls
    )

    # Stream back the file
    def iterfile():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
        os.unlink(video_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")

# ---------- Subtitling ----------
def add_subtitles_to_video(video_path, script_data):
    """Burn subtitles into the video."""
    # We'll use moviepy TextClip and CompositeVideoClip
    clip = VideoFileClip(video_path)
    chapters = script_data.get("chapters", [])
    # For simplicity, we'll split the total duration equally among chapters and place titles
    # In a real implementation, you'd align with TTS timestamps.
    # Here we'll just overlay chapter titles as text at staggered intervals.
    overlay_clips = []
    total_dur = clip.duration
    for i, ch in enumerate(chapters):
        start_time = i * total_dur / len(chapters)
        txt_clip = TextClip(
            text=ch.get("title", "").upper(),
            font="Montserrat-ExtraBold",  # requires font file, fallback to default
            font_size=48,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(int(clip.w*0.9), None),
        ).set_position("center").set_start(start_time).set_duration(4)
        overlay_clips.append(txt_clip)

    final = CompositeVideoClip([clip] + overlay_clips)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out:
        final.write_videofile(out.name, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast")
        out_path = out.name
    clip.close()
    return out_path

@app.post("/api/add-subtitles")
async def add_subtitles(
    video: UploadFile = File(...),
    script: str = Form(...),
):
    script_data = json.loads(script)
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        input_path = tmp.name

    loop = asyncio.get_running_loop()
    output_path = await loop.run_in_executor(
        None, add_subtitles_to_video, input_path, script_data
    )
    os.unlink(input_path)

    def iterfile():
        with open(output_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
        os.unlink(output_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")

# ---------- Health Check ----------
@app.get("/")
async def root():
    return {"status": "Curiosity Trap Backend running"}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port)