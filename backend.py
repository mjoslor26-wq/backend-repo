import os
import io
import json
import tempfile
import asyncio
import logging
from typing import List, Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import google.generativeai as genai

# MoviePy imports (v2+ style)
from moviepy import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
    vfx,
)

# For TTS (free & realistic)
import edge_tts

# ----------------------------------------------------------------------
# Logging & config
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("curiosity-trap-backend")

app = FastAPI(title="Curiosity Trap Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Helper: call Gemini to create a documentary script
# ----------------------------------------------------------------------
async def generate_script(theme: str, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Create a 8‑minute documentary script about "{theme}". 
    The output must be a **JSON object** with these keys:
    - title (string)
    - chapters (list of objects, each with "title" and "text" (the narration))
    - fullText (string)  -- the complete narration merged from all chapters
    - keyEntities (list of strings) -- main names/places/terms
    """
    response = model.generate_content(prompt)
    # Gemini may wrap in ```json, so strip markers
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse Gemini script output")
    return data

# ----------------------------------------------------------------------
# Helper: extract keywords from script (already done on frontend,
# but we need it here too for image search)
# ----------------------------------------------------------------------
def extract_keywords(script_data: dict, top_n: int = 20) -> List[str]:
    keywords = set()
    for ch in script_data.get("chapters", []):
        for w in ch["title"].split():
            if len(w) > 3:
                keywords.add(w.lower())
    title_words = script_data.get("title", "").split()
    for w in title_words:
        if len(w) > 3:
            keywords.add(w.lower())
    for ent in script_data.get("keyEntities", []):
        keywords.add(ent.lower())
    return list(keywords)[:top_n]

# ----------------------------------------------------------------------
# Helper: fetch images from Pexels
# ----------------------------------------------------------------------
async def fetch_pexels_images(keywords: List[str], api_key: str, count: int = 15) -> List[str]:
    headers = {"Authorization": api_key}
    urls = []
    for kw in keywords:
        if len(urls) >= count:
            break
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params={"query": kw, "per_page": min(3, count - len(urls))},
        )
        if resp.status_code == 200:
            data = resp.json()
            for photo in data.get("photos", []):
                urls.append(photo["src"]["large"])
    return urls[:count]

# ----------------------------------------------------------------------
# Helper: generate word‑level subtitles (approximate)
# ----------------------------------------------------------------------
def generate_word_subtitles(full_text: str, duration: float) -> List[dict]:
    words = full_text.split()
    if not words:
        return []
    word_duration = duration / len(words)
    subs = []
    for i, word in enumerate(words):
        start = i * word_duration
        end = start + word_duration
        subs.append({"start": start, "end": end, "text": word})
    return subs

# ----------------------------------------------------------------------
# Helper: create video with Ken Burns effect + audio + subtitles
# ----------------------------------------------------------------------
async def build_video(script_data: dict, audio_path: str, image_urls: List[str], output_path: str):
    # Load audio to get duration
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration

    # Prepare word subtitles
    word_subs = generate_word_subtitles(script_data.get("fullText", ""), total_duration)

    # Create image clips with Ken Burns (zoom effect)
    image_clips = []
    clip_duration = total_duration / len(image_urls) if image_urls else total_duration

    for url in image_urls:
        # Download image to temp file
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            continue
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(r.content)
            img_path = f.name

        clip = ImageClip(img_path).with_duration(clip_duration)
        # Apply Ken Burns: slow zoom in or out
        zoom = vfx.Resize(lambda t: 1 + 0.1 * (t / clip_duration))
        clip = clip.with_effects([zoom])
        # Crossfade could be added here, but simplify
        image_clips.append(clip)

    if not image_clips:
        raise HTTPException(500, "No valid images found")

    # Concatenate image clips (the import we fixed!)
    video = concatenate_videoclips(image_clips, method="compose")
    video = video.with_audio(audio_clip)

    # Burn word subtitles
    subtitle_clips = []
    for sub in word_subs:
        txt = TextClip(
            text=sub["text"],
            font_size=48,
            color="white",
            stroke_color="black",
            stroke_width=2,
            font="Arial",
        )
        txt = txt.with_position(("center", "center")).with_start(sub["start"]).with_duration(
            sub["end"] - sub["start"]
        )
        subtitle_clips.append(txt)

    final = CompositeVideoClip([video] + subtitle_clips)

    final.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )

    # Cleanup image temp files
    for clip in image_clips:
        if hasattr(clip, "filename"):
            try:
                os.unlink(clip.filename)
            except:
                pass

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/api/generate-script")
async def api_generate_script(data: dict):
    theme = data.get("theme")
    api_key = data.get("apiKey")
    if not theme or not api_key:
        raise HTTPException(400, "theme and apiKey are required")
    try:
        script = await generate_script(theme, api_key)
        return script
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/generate-tts")
async def api_generate_tts(data: dict):
    text = data.get("text")
    voice = data.get("voice", "en-US-ChristopherNeural")
    if not text:
        raise HTTPException(400, "text is required")

    # Use Edge TTS to generate audio in memory
    communicate = edge_tts.Communicate(text, voice)
    mp3_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data.write(chunk["data"])
    mp3_data.seek(0)

    return StreamingResponse(
        mp3_data,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=tts.mp3"},
    )

@app.post("/api/fetch-images")
async def api_fetch_images(data: dict):
    keywords = data.get("keywords")
    api_key = data.get("apiKey")
    count = data.get("count", 15)
    if not keywords or not api_key:
        raise HTTPException(400, "keywords and apiKey are required")
    try:
        urls = await fetch_pexels_images(keywords, api_key, count)
        return {"urls": urls}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/compose-video")
async def api_compose_video(
    script: str = Form(...),
    audio: UploadFile = File(...),
    imageUrls: str = Form(...),
    duration: Optional[str] = Form("480"),
):
    script_data = json.loads(script)
    image_urls = json.loads(imageUrls)

    # Save audio to a temporary file
    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    # Output video file
    output_path = tempfile.mktemp(suffix=".mp4")

    try:
        await build_video(script_data, audio_path, image_urls, output_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="documentary.mp4",
        background=None,  # Removes the file after response sent (FastAPI 0.111+)
    )

@app.post("/api/add-subtitles")
async def api_add_subtitles(
    video: UploadFile = File(...),
    script: str = Form(...),
):
    script_data = json.loads(script)

    # Save uploaded video to a temp file
    video_bytes = await video.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        input_path = f.name

    # Load video + audio
    clip = VideoFileClip(input_path)
    audio = clip.audio
    duration = clip.duration

    word_subs = generate_word_subtitles(script_data.get("fullText", ""), duration)

    subtitles = []
    for sub in word_subs:
        txt = TextClip(
            text=sub["text"],
            font_size=48,
            color="white",
            stroke_color="black",
            stroke_width=2,
            font="Arial",
        )
        txt = txt.with_position(("center", "center")).with_start(sub["start"]).with_duration(
            sub["end"] - sub["start"]
        )
        subtitles.append(txt)

    final = CompositeVideoClip([clip] + subtitles)
    output_path = tempfile.mktemp(suffix=".mp4")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )

    # Cleanup
    os.unlink(input_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="documentary_subtitled.mp4",
    )

# ----------------------------------------------------------------------
# Run with: uvicorn app:app --host 0.0.0.0 --port 10000
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
