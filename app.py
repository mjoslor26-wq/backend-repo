# app.py - ULTRA LIGHT (Free tier safe) - 5min video, 480p, 12 segments

import os
import re
import asyncio
import aiohttp
import uuid
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

from gtts import gTTS
import nltk
import numpy as np
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, vfx

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY', '')
OUTPUT_DIR = Path("/tmp/generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Auto Video Generator")

JOBS_DIR = OUTPUT_DIR / "jobs_meta"
JOBS_DIR.mkdir(exist_ok=True)

def save_job(job_id: str, job_data: dict):
    with open(JOBS_DIR / f"{job_id}.json", 'w') as f:
        json.dump(job_data, f)

def load_job(job_id: str):
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def update_job(job_id: str, updates: dict):
    job = load_job(job_id)
    if job:
        job.update(updates)
        save_job(job_id, job)

def get_job(job_id: str):
    return load_job(job_id)

# ------------------- HTML TEMPLATES (same, but with better error handling) -------------------
LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Video Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 32px;
            padding: 48px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .subtitle { color: #4b5563; margin: 12px 0 32px; }
        .features { display: grid; grid-template-columns: repeat(2,1fr); gap: 12px; margin: 32px 0; }
        .feature { background: #f3f4f6; padding: 10px; border-radius: 16px; text-align: center; }
        input {
            width: 100%;
            padding: 16px;
            font-size: 1.1em;
            border: 2px solid #e5e7eb;
            border-radius: 24px;
            margin-bottom: 24px;
        }
        button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 16px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 40px;
            cursor: pointer;
            width: 100%;
        }
        .loader { display: none; margin-top: 20px; text-align: center; }
        .spinner {
            border: 3px solid #e5e7eb;
            border-top: 3px solid #8b5cf6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">
    <h1>🎬 AI Video Generator</h1>
    <p class="subtitle">Create a 5‑minute documentary from any theme</p>
    <div class="features">
        <div class="feature">🎥 5-min Video</div>
        <div class="feature">📱 2 Shorts</div>
        <div class="feature">🖼️ Thumbnail</div>
        <div class="feature">📝 Metadata</div>
    </div>
    <form id="generateForm">
        <input type="text" name="theme" placeholder="e.g., 'Ancient Rome', 'Space Travel'" required>
        <button type="submit">🚀 Generate Video</button>
    </form>
    <div class="loader" id="loader">
        <div class="spinner"></div>
        <p style="margin-top: 12px;">Processing... about 4‑5 minutes.</p>
    </div>
</div>
<script>
    document.getElementById('generateForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const loader = document.getElementById('loader');
        const btn = e.target.querySelector('button');
        loader.style.display = 'block';
        btn.disabled = true;
        try {
            const res = await fetch('/generate', { method: 'POST', body: formData });
            const data = await res.json();
            window.location.href = `/result/${data.job_id}`;
        } catch(err) {
            alert('Error: ' + err.message);
            loader.style.display = 'none';
            btn.disabled = false;
        }
    });
</script>
</body>
</html>
"""

RESULT_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Video is Ready</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui; background: #0f172a; padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 32px; padding: 32px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; flex-wrap: wrap; }
        .new-btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 10px 24px; border-radius: 40px; text-decoration: none; }
        .progress-section { text-align: center; padding: 60px 20px; }
        .progress-bar { width: 100%; height: 28px; background: #e2e8f0; border-radius: 14px; overflow: hidden; margin: 24px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
        .results { display: none; }
        .video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin: 32px 0; }
        .card { background: #f8fafc; border-radius: 24px; padding: 24px; }
        video, img { width: 100%; border-radius: 16px; margin: 16px 0; background: #000; }
        button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 40px; cursor: pointer; }
        .metadata { background: #f1f5f9; border-radius: 24px; padding: 24px; margin-top: 32px; }
        .title-box, .desc-box { background: white; padding: 16px; border-radius: 16px; margin: 12px 0; }
        .short-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .error-box { background: #fee2e2; border: 1px solid #f87171; border-radius: 16px; padding: 32px; text-align: center; margin: 40px; }
        @media (max-width: 768px) { .video-grid, .short-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>✨ Video Studio</h1>
        <a href="/" class="new-btn">+ New Video</a>
    </div>
    <div id="errorBox" class="error-box" style="display: none;">
        <h2>❌ Generation Failed</h2>
        <p>The server ran out of memory. Please try a shorter theme or upgrade Render to Basic plan.</p>
        <a href="/" style="background: #3b82f6; color: white; padding: 10px 20px; border-radius: 40px; text-decoration: none; display: inline-block; margin-top: 16px;">Try Again</a>
    </div>
    <div id="progressSection" class="progress-section">
        <h2>🔄 Generating...</h2>
        <div class="progress-bar"><div class="progress-fill" id="progressFill">0%</div></div>
        <p id="statusMsg">Starting...</p>
        <div>⏳ ~4-5 minutes</div>
    </div>
    <div id="resultsSection" class="results">
        <h2>🎉 Ready!</h2>
        <div class="video-grid">
            <div class="card">
                <h3>🎥 Main Video (5 min)</h3>
                <video controls><source src="" type="video/mp4"></video>
                <button onclick="downloadFile('video')">Download</button>
            </div>
            <div class="card">
                <h3>🖼️ Thumbnail</h3>
                <img id="thumbnailImg" src="">
                <button onclick="downloadFile('thumbnail')">Download</button>
            </div>
        </div>
        <div class="metadata">
            <h3>📝 Metadata</h3>
            <div class="title-box" id="titleDisplay"></div>
            <div class="desc-box" id="descDisplay"></div>
            <button onclick="copyMetadata()">Copy</button>
        </div>
        <div class="card">
            <h3>📱 Shorts</h3>
            <div class="short-row">
                <div><video controls><source src="" type="video/mp4"></video><button onclick="downloadFile('short1')">Short 1</button></div>
                <div><video controls><source src="" type="video/mp4"></video><button onclick="downloadFile('short2')">Short 2</button></div>
            </div>
        </div>
    </div>
</div>
<script>
    const jobId = "{{ job_id }}";
    let interval = setInterval(checkStatus, 2000);
    let consecutiveErrors = 0;
    async function checkStatus() {
        try {
            const res = await fetch(`/status/${jobId}`);
            if (res.status === 502) {
                consecutiveErrors++;
                if (consecutiveErrors > 3) {
                    clearInterval(interval);
                    document.getElementById('progressSection').style.display = 'none';
                    document.getElementById('errorBox').style.display = 'block';
                }
                return;
            }
            if (res.status === 404) {
                clearInterval(interval);
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('errorBox').style.display = 'block';
                return;
            }
            const data = await res.json();
            consecutiveErrors = 0;
            document.getElementById('progressFill').style.width = data.progress + '%';
            document.getElementById('progressFill').innerText = data.progress + '%';
            document.getElementById('statusMsg').innerText = data.status || 'Processing...';
            if (data.completed) {
                clearInterval(interval);
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('resultsSection').style.display = 'block';
                loadAssets();
            } else if (data.error && data.error.includes('Memory')) {
                clearInterval(interval);
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('errorBox').style.display = 'block';
            }
        } catch(e) {
            console.error(e);
            consecutiveErrors++;
            if (consecutiveErrors > 3) {
                clearInterval(interval);
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('errorBox').style.display = 'block';
            }
        }
    }
    function loadAssets() {
        const videos = document.querySelectorAll('video');
        videos[0].querySelector('source').src = `/download/${jobId}/video`;
        videos[0].load();
        videos[1].querySelector('source').src = `/download/${jobId}/short1`;
        videos[1].load();
        videos[2].querySelector('source').src = `/download/${jobId}/short2`;
        videos[2].load();
        document.getElementById('thumbnailImg').src = `/download/${jobId}/thumbnail`;
        fetch(`/status/${jobId}`).then(r=>r.json()).then(data=>{
            if (!data.error) {
                document.getElementById('titleDisplay').innerText = data.title || 'Your Title';
                document.getElementById('descDisplay').innerText = data.description || 'Description';
            }
        });
    }
    function downloadFile(type) { window.location.href = `/download/${jobId}/${type}`; }
    function copyMetadata() {
        const title = document.getElementById('titleDisplay').innerText;
        const desc = document.getElementById('descDisplay').innerText;
        navigator.clipboard.writeText(`Title: ${title}\\n\\nDescription:\\n${desc}`).then(()=>alert('Copied!'));
    }
</script>
</body>
</html>
"""

# ------------------- Ultra Light Generator -------------------
class VideoGenerator:
    def __init__(self, theme: str, job_id: str):
        self.theme = theme
        self.job_id = job_id
        self.job_dir = OUTPUT_DIR / job_id
        self.job_dir.mkdir(exist_ok=True)
        self.audio_dir = self.job_dir / "audio"
        self.images_dir = self.job_dir / "images"
        self.audio_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def update_status(self, message: str, progress: int):
        update_job(self.job_id, {'status': message, 'progress': progress})

    async def fetch_content(self) -> str:
        self.update_status("Researching...", 5)
        try:
            import wikipedia
            search = wikipedia.search(self.theme, results=1)
            if not search:
                return f"Explore {self.theme}. This guide covers everything you need."
            page = wikipedia.page(search[0])
            return page.summary[:800] + f" Let's explore {self.theme} in detail."
        except:
            return f"{self.theme} is fascinating. Learn all about it in this short documentary."

    async def generate_tts(self, text: str, output_path: Path) -> float:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._synth, text, output_path)
        audio = AudioFileClip(str(output_path))
        dur = audio.duration
        audio.close()
        return dur

    def _synth(self, text: str, path: Path):
        gTTS(text=text, lang='en', slow=False).save(str(path))

    async def fetch_image(self, query: str, idx: int) -> Path:
        img_path = self.images_dir / f"img_{idx}.jpg"
        if UNSPLASH_ACCESS_KEY:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.unsplash.com/search/photos?query={query}&per_page=1&orientation=landscape"
                    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
                    async with session.get(url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data['results']:
                                img_url = data['results'][0]['urls']['small']  # smaller image
                                async with session.get(img_url) as img_resp:
                                    img_data = await img_resp.read()
                                    with open(img_path, 'wb') as f:
                                        f.write(img_data)
                                    return img_path
            except:
                pass
        # Placeholder 640x360 (very small)
        img = Image.new('RGB', (640, 360), color=(25, 25, 45))
        draw = ImageDraw.Draw(img)
        draw.text((320,180), self.theme[:30], fill=(255,215,0), anchor="mm")
        img.save(img_path)
        return img_path

    async def create_clip(self, img_path: Path, audio_path: Path):
        audio = AudioFileClip(str(audio_path))
        dur = audio.duration
        clip = ImageClip(str(img_path)).resize(height=360).set_duration(dur)
        clip = clip.set_audio(audio)
        return clip.fx(vfx.fadein, 0.2).fx(vfx.fadeout, 0.2)

    async def generate(self):
        try:
            content = await self.fetch_content()
            # Split into 12 segments max
            sentences = re.split(r'(?<=[.!?])\s+', content)
            segments = []
            current = []
            for sent in sentences[:40]:
                if len(' '.join(current + [sent])) > 300:
                    if current:
                        segments.append(' '.join(current))
                    current = [sent]
                else:
                    current.append(sent)
            if current:
                segments.append(' '.join(current))
            segments = segments[:12]  # max 12 clips ~ 4-5 min video
            while len(segments) < 6:
                segments.append(f"More about {self.theme}...")
            clips = []
            total = len(segments)
            for i, seg in enumerate(segments):
                self.update_status(f"Scene {i+1}/{total}", 20 + int((i/total)*60))
                audio_path = self.audio_dir / f"aud_{i}.mp3"
                await self.generate_tts(seg, audio_path)
                kw = seg.split()[:3]
                img_path = await self.fetch_image(' '.join(kw), i)
                clip = await self.create_clip(img_path, audio_path)
                clips.append(clip)
                gc.collect()
            self.update_status("Rendering video...", 85)
            final = concatenate_videoclips(clips, method="compose")
            out_video = self.job_dir / "final.mp4"
            final.write_videofile(
                str(out_video), fps=20, codec='libx264', audio_codec='aac',
                preset='ultrafast', bitrate='500k', threads=1
            )
            final.close()
            for c in clips:
                c.close()
            gc.collect()
            # Shorts
            from moviepy.video.io.VideoFileClip import VideoFileClip
            vid = VideoFileClip(str(out_video))
            dur = vid.duration
            clip1 = vid.subclip(0, min(60, dur))
            clip1.write_videofile(str(self.job_dir / "short1.mp4"), preset='ultrafast')
            start = max(0, dur-60) if dur>120 else max(0, dur-60)
            clip2 = vid.subclip(start, min(start+60, dur))
            clip2.write_videofile(str(self.job_dir / "short2.mp4"), preset='ultrafast')
            vid.close()
            thumbnail = await self.make_thumbnail()
            title = f"{self.theme.upper()} • 5-Min Guide"
            desc = f"🎬 {self.theme} - quick documentary.\nLike and subscribe!"
            update_job(self.job_id, {
                'completed': True,
                'video_path': str(out_video),
                'short1_path': str(self.job_dir / "short1.mp4"),
                'short2_path': str(self.job_dir / "short2.mp4"),
                'thumbnail_path': str(thumbnail),
                'title': title,
                'description': desc
            })
            self.update_status("Done!", 100)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            update_job(self.job_id, {'status': error_msg, 'error': str(e)})

    async def make_thumbnail(self) -> Path:
        img = Image.new('RGB', (1280,720), color=(20,20,40))
        draw = ImageDraw.Draw(img)
        draw.text((640,360), self.theme.upper(), fill=(255,215,0), anchor="mm")
        path = self.job_dir / "thumb.jpg"
        img.save(path)
        return path

# ------------------- Routes -------------------
@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse(content=LANDING_PAGE_HTML)

@app.post("/generate")
async def generate(theme: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    job_id = str(uuid.uuid4())
    job_data = {'status': 'Starting', 'progress': 0, 'completed': False, 'theme': theme}
    save_job(job_id, job_data)
    gen = VideoGenerator(theme, job_id)
    background_tasks.add_task(gen.generate)
    return JSONResponse({'job_id': job_id})

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404)
    return JSONResponse(job)

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def result(job_id: str):
    html = RESULT_PAGE_HTML.replace("{{ job_id }}", job_id)
    return HTMLResponse(content=html)

@app.get("/download/{job_id}/{file_type}")
async def download(job_id: str, file_type: str):
    job = get_job(job_id)
    if not job or not job.get('completed'):
        raise HTTPException(404)
    mapping = {'video': 'video_path', 'short1': 'short1_path', 'short2': 'short2_path', 'thumbnail': 'thumbnail_path'}
    path = job.get(mapping.get(file_type))
    if not path or not Path(path).exists():
        raise HTTPException(404)
    return FileResponse(path, filename=Path(path).name)

if __name__ == "__main__":
    import uvicorn
    print("Ultra-light version running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
