#!/usr/bin/env python3
"""
Curiosity Trap – Video Generation Backend
Fallback system:
- Tries multiple Gemini API keys (from frontend) before switching models
- Model order: best first → reliable high‑quota fallbacks
- Polling TTS, Ken Burns video, subtitles
"""

import os, json, io, asyncio, tempfile, logging, uuid, time
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from google import genai
from google.genai import types as genai_types

import edge_tts

from moviepy import (
    VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
    TextClip, concatenate_videoclips, vfx
)
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("curiosity-backend")

app = FastAPI(title="Curiosity Trap Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PEXELS_API_URL = "https://api.pexels.com/v1/search"

# Model fallback order – best models first, then high‑quota ones
SCRIPT_MODELS = [
    "gemini-2.5-flash",      # High quality, moderate quota
    "gemini-3-flash",        # Newer generation, may have limited quota
    "gemini-2.0-flash",      # King of free tier (1,500 req/day) – reliable backup
    "gemini-2.5-flash-lite", # Lightweight, generous limits
]

tts_tasks: Dict[str, Dict] = {}

# ----------------------------------------------------------------------
# 1. SCRIPT GENERATION with multi‑key + model fallback
# ----------------------------------------------------------------------
def generate_script(theme: str, api_keys: List[str]) -> dict:
    """Try each model, and for each model try each API key."""
    last_error_msg = ""

    for model_name in SCRIPT_MODELS:
        logger.info(f"Trying model: {model_name}")
        for key_idx, api_key in enumerate(api_keys):
            if not api_key:
                continue
            client = genai.Client(api_key=api_key)
            for attempt in range(3):  # retry up to 3 times on 503 / 429
                try:
                    prompt = f"""
You are a script writer for the documentary series "The Curiosity Trap". Write a script for an EXACTLY 8‑minute video (about 900‑1000 words) on the theme: "{theme}".

Structure:
- Title: bold and provocative.
- Hook: a gap in knowledge, a void effect.
- 6–8 short chapters, each with a suggestive one-line title.
- For each chapter, 5‑7 sentences of narration (under 12 words per sentence).
- Use dramatic adjectives, personification, contrasts.
- End with an open reflection or warning.
- Include a list of key entities (people, places, things).

Output ONLY a valid JSON object with these exact keys:
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
The fullText MUST be at least 900 words long.
Return only valid JSON, no other text.
"""
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                        ),
                    )
                    raw = response.text.strip()
                    if raw.startswith("```json"):
                        raw = raw[7:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()
                    script = json.loads(raw)

                    # Ensure fullText exists
                    if "fullText" not in script:
                        full = ""
                        for ch in script.get("chapters", []):
                            full += f"Chapter: {ch['title']}. {ch['text']} "
                        script["fullText"] = full.strip()

                    word_count = len(script.get("fullText", "").split())
                    logger.info(f"Script word count: {word_count}")
                    if word_count < 800:
                        logger.warning(f"Script too short ({word_count} words). Trying next key/model.")
                        break  # exit retry loop for this key
                    return script

                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"Model {model_name}, key #{key_idx+1}, attempt {attempt+1}: {err_str}")
                    if "429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                        # Temporary overload or quota – wait and retry
                        if attempt < 2:
                            time.sleep(10)
                            continue
                    # For other errors (or final attempt), move to next key
                    break
        # If we exhaust all keys for this model, try next model (loop continues)
        last_error_msg = f"Model {model_name} failed with all provided keys."

    raise HTTPException(
        429,
        f"All models and keys exhausted. Last error: {last_error_msg} Please wait or add a new Gemini API key.",
    )

# ----------------------------------------------------------------------
# 2. POLLING TTS (unchanged)
# ----------------------------------------------------------------------
def _run_tts_generation(task_id: str, text: str):
    try:
        communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(communicate.save(tmp.name))
            loop.close()
            tts_tasks[task_id] = {"status": "completed", "file": tmp.name}
        logger.info(f"TTS task {task_id} completed.")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        tts_tasks[task_id] = {"status": "error", "error": str(e)}

@app.post("/api/generate-tts")
async def api_start_tts(data: dict, background_tasks: BackgroundTasks):
    text = data.get("text")
    if not text: raise HTTPException(400, "text required")
    task_id = uuid.uuid4().hex
    tts_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(_run_tts_generation, task_id, text)
    return {"task_id": task_id, "status": "processing"}

@app.get("/api/tts-status/{task_id}")
async def tts_status(task_id: str):
    task = tts_tasks.get(task_id)
    if not task: raise HTTPException(404, "Task not found")
    return {"status": task["status"]}

@app.get("/api/tts-download/{task_id}")
async def tts_download(task_id: str):
    task = tts_tasks.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(404, "File not ready")
    return FileResponse(task["file"], media_type="audio/mpeg")

# ----------------------------------------------------------------------
# 3. HELPERS (images, video, subtitles) – unchanged
# ----------------------------------------------------------------------
def extract_keywords(script_data: dict) -> List[str]:
    keywords = set()
    for ch in script_data.get("chapters", []):
        for w in ch["title"].split():
            if len(w) > 3: keywords.add(w.lower())
    for w in script_data.get("title", "").split():
        if len(w) > 3: keywords.add(w.lower())
    for ent in script_data.get("keyEntities", []):
        keywords.add(ent.lower())
    return list(keywords)[:20]

def fetch_images(keywords: List[str], api_key: str, count: int = 15) -> List[str]:
    headers = {"Authorization": api_key}
    urls = []
    for kw in keywords[:5]:
        if len(urls) >= count: break
        params = {"query": kw, "per_page": min(3, count - len(urls)), "orientation": "landscape"}
        resp = requests.get(PEXELS_API_URL, headers=headers, params=params)
        if resp.status_code == 200:
            for photo in resp.json().get("photos", []):
                urls.append(photo["src"]["large"])
    return urls[:count]

def download_image(url, max_size=1920):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raw.decode_content = True
    img = Image.open(resp.raw).convert("RGB")
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
    return np.array(img)

def apply_vignette(frame, intensity=0.6):
    h, w = frame.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    mask = 1 - np.sqrt(X**2 + Y**2) * 0.9
    mask = np.clip(mask, 0, 1) ** 1.5
    mask = 1 - intensity + intensity * mask
    return (frame * mask[..., np.newaxis]).astype(np.uint8)

def build_video(script_data, audio_bytes, image_urls):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    audio_clip = AudioFileClip(audio_path)
    total_dur = audio_clip.duration
    images = []
    for url in image_urls:
        try: images.append(download_image(url))
        except: continue
    if not images: raise ValueError("No valid images")
    avg_dur = total_dur / len(images)
    clips = []
    for img in images:
        clip = ImageClip(img).with_duration(avg_dur)
        clip = clip.resized(lambda t: 1.0 + 0.05 * (t / avg_dur) if avg_dur > 0 else 1.0)
        clip = clip.transform(lambda frame: apply_vignette(frame))
        clips.append(clip)
    video = concatenate_videoclips(clips, method="compose")
    if video.duration < total_dur:
        loops = int(np.ceil(total_dur / video.duration))
        video = concatenate_videoclips([video] * loops, method="compose")
    video = video.subclip(0, total_dur).with_audio(audio_clip)
    output_path = tempfile.mktemp(suffix=".mp4")
    video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac",
                          threads=2, preset="ultrafast", logger=None, verbose=False)
    audio_clip.close(); video.close()
    os.unlink(audio_path)
    return output_path

# ----------------------------------------------------------------------
# 4. ENDPOINTS
# ----------------------------------------------------------------------
@app.post("/api/generate-script")
async def api_generate_script(data: dict):
    theme = data.get("theme")
    # Accept either a single key or a list of keys
    api_keys_input = data.get("apiKeys")  # frontend can send a list
    if not api_keys_input:
        # fallback to single key for backward compatibility
        single_key = data.get("apiKey")
        if single_key:
            api_keys_input = [single_key]
        else:
            raise HTTPException(400, "At least one Gemini API key is required")
    if isinstance(api_keys_input, str):
        api_keys_input = [api_keys_input]
    if not theme:
        raise HTTPException(400, "theme is required")
    return generate_script(theme, api_keys_input)

@app.post("/api/fetch-images")
async def api_fetch_images(data: dict):
    keywords = data.get("keywords", [])
    api_key = data.get("apiKey")
    count = data.get("count", 15)
    if not keywords or not api_key: raise HTTPException(400, "keywords and apiKey required")
    return {"urls": fetch_images(keywords, api_key, count)}

@app.post("/api/compose-video")
async def api_compose_video(
    script: str = Form(...),
    audio: UploadFile = File(...),
    imageUrls: str = Form(...),
):
    script_data = json.loads(script)
    image_urls = json.loads(imageUrls)
    audio_bytes = await audio.read()
    loop = asyncio.get_running_loop()
    video_path = await loop.run_in_executor(None, build_video, script_data, audio_bytes, image_urls)
    def iterfile():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024*1024): yield chunk
        os.unlink(video_path)
    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.post("/api/add-subtitles")
async def api_add_subtitles(
    video: UploadFile = File(...),
    script: str = Form(...),
):
    script_data = json.loads(script)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        input_path = tmp.name
    clip = VideoFileClip(input_path)
    total_dur = clip.duration
    words = script_data.get("fullText", "").split()
    if not words: raise HTTPException(400, "No text")
    word_dur = total_dur / len(words)
    txt_clips = []
    for i, word in enumerate(words):
        txt = TextClip(text=word, font_size=48, color="white", stroke_color="black",
                       stroke_width=2, font="Arial")
        txt = txt.with_position(("center","center")).with_start(i*word_dur).with_duration(word_dur)
        txt_clips.append(txt)
    final = CompositeVideoClip([clip] + txt_clips)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out:
        final.write_videofile(out.name, fps=24, codec="libx264", audio_codec="aac",
                              preset="ultrafast", verbose=False, logger=None)
        output_path = out.name
    clip.close(); final.close()
    os.unlink(input_path)
    def iterfile():
        with open(output_path, "rb") as f:
            while chunk := f.read(1024*1024): yield chunk
        os.unlink(output_path)
    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.get("/")
async def root():
    return {"status": "Curiosity Trap Backend running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
