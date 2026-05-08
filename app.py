# app.py - Complete AI Video Generation System with UI (using gTTS)
# Deployable on Render.com - includes PIL compatibility fix

import os
import re
import asyncio
import aiohttp
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont

# >>> FIX for Pillow >= 10.0.0: ANTIALIAS removed, replace with LANCZOS
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

from gtts import gTTS
import nltk
import numpy as np
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, vfx

# Download NLTK data (will happen on first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Configuration
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY', '')
OUTPUT_DIR = Path("/tmp/generated_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Auto Video Generator")
jobs: Dict[str, dict] = {}

TARGET_DURATION = 480  # 8 minutes
MAX_SHORT_DURATION = 60
WORDS_PER_MINUTE = 140

# ----------------------------------------------------------------------
# HTML Templates (embedded - same as before)
# ----------------------------------------------------------------------

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Video Generator • Create Viral Videos</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: rgba(255,255,255,0.98);
            border-radius: 32px;
            padding: 48px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        }
        h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 12px;
        }
        .subtitle { color: #4b5563; font-size: 1.1em; margin-bottom: 32px; }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin: 32px 0;
        }
        .feature {
            background: #f3f4f6;
            padding: 12px;
            border-radius: 16px;
            font-weight: 500;
            text-align: center;
        }
        input {
            width: 100%;
            padding: 16px 20px;
            font-size: 1.1em;
            border: 2px solid #e5e7eb;
            border-radius: 24px;
            margin-bottom: 24px;
        }
        input:focus {
            outline: none;
            border-color: #8b5cf6;
        }
        button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 16px 32px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 40px;
            cursor: pointer;
            width: 100%;
        }
        button:hover { transform: translateY(-2px); }
        .info {
            background: #f9fafb;
            border-radius: 20px;
            padding: 20px;
            margin-top: 32px;
            font-size: 0.9em;
            color: #6b7280;
        }
        .loader { display: none; margin-top: 24px; text-align: center; }
        .spinner {
            border: 3px solid #e5e7eb;
            border-top: 3px solid #8b5cf6;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 AI Video Generator</h1>
        <p class="subtitle">Turn any idea into a complete YouTube-ready video</p>
        <div class="features">
            <div class="feature">🎥 8-min Documentary</div>
            <div class="feature">📱 2 TikTok Shorts</div>
            <div class="feature">🖼️ Clickable Thumbnail</div>
            <div class="feature">📝 SEO Title & Desc</div>
        </div>
        <form id="generateForm">
            <input type="text" name="theme" placeholder="e.g., 'Quantum Computing', 'Ancient Egypt', 'Electric Cars'" required autocomplete="off">
            <button type="submit">🔥 Generate Video Now</button>
        </form>
        <div class="loader" id="loader">
            <div class="spinner"></div>
            <p style="margin-top: 12px;">AI is crafting your video... (3-5 min)</p>
        </div>
        <div class="info">
            💡 <strong>Pro tip:</strong> Use specific themes for best results. Video includes voiceover, background music, transitions, and auto-generated clips.
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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Video is Ready!</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: #0f172a; padding: 24px; }
        .container { max-width: 1400px; margin: 0 auto; background: white; border-radius: 32px; padding: 32px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; flex-wrap: wrap; }
        .new-btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 10px 24px; border-radius: 40px; text-decoration: none; font-weight: 600; }
        .progress-section { text-align: center; padding: 60px 20px; }
        .progress-bar { width: 100%; height: 28px; background: #e2e8f0; border-radius: 14px; overflow: hidden; margin: 24px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
        .results { display: none; }
        .video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin: 32px 0; }
        .card { background: #f8fafc; border-radius: 24px; padding: 24px; }
        video, img { width: 100%; border-radius: 16px; margin: 16px 0; background: #000; }
        button, .download-btn { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 40px; cursor: pointer; margin: 4px; display: inline-block; }
        .metadata { background: #f1f5f9; border-radius: 24px; padding: 24px; margin-top: 32px; }
        .title-box { background: white; padding: 16px; border-radius: 16px; font-weight: bold; margin: 12px 0; }
        .desc-box { background: white; padding: 16px; border-radius: 16px; font-family: monospace; white-space: pre-wrap; font-size: 0.9em; }
        .short-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        @media (max-width: 768px) { .video-grid, .short-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>✨ Video Studio</h1>
        <a href="/" class="new-btn">+ New Video</a>
    </div>
    <div id="progressSection" class="progress-section">
        <h2>🔄 Generating your content...</h2>
        <div class="progress-bar"><div class="progress-fill" id="progressFill">0%</div></div>
        <p id="statusMsg" style="color: #475569;">Starting up...</p>
        <div>⏳ This takes ~3-5 minutes. Please wait.</div>
    </div>
    <div id="resultsSection" class="results">
        <h2>🎉 Your video package is ready!</h2>
        <div class="video-grid">
            <div class="card">
                <h3>🎥 Main 8‑min Video</h3>
                <video controls><source src="" type="video/mp4"></video>
                <button onclick="downloadFile('video')">⬇️ Download MP4</button>
            </div>
            <div class="card">
                <h3>🖼️ YouTube Thumbnail</h3>
                <img id="thumbnailImg" src="" alt="thumbnail">
                <button onclick="downloadFile('thumbnail')">⬇️ Download JPG</button>
            </div>
        </div>
        <div class="metadata">
            <h3>📈 Optimized YouTube Metadata</h3>
            <div class="title-box" id="titleDisplay"></div>
            <div class="desc-box" id="descDisplay"></div>
            <button onclick="copyMetadata()">📋 Copy Title & Description</button>
        </div>
        <div class="card">
            <h3>📱 TikTok / Shorts (1 min each)</h3>
            <div class="short-row">
                <div><video controls><source src="" type="video/mp4"></video><button onclick="downloadFile('short1')">Download Short #1</button></div>
                <div><video controls><source src="" type="video/mp4"></video><button onclick="downloadFile('short2')">Download Short #2</button></div>
            </div>
        </div>
    </div>
</div>
<script>
    const jobId = "{{ job_id }}";
    let interval = setInterval(checkStatus, 2000);
    async function checkStatus() {
        try {
            const res = await fetch(`/status/${jobId}`);
            const data = await res.json();
            document.getElementById('progressFill').style.width = data.progress + '%';
            document.getElementById('progressFill').innerText = data.progress + '%';
            document.getElementById('statusMsg').innerText = data.status || 'Processing...';
            if (data.completed) {
                clearInterval(interval);
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('resultsSection').style.display = 'block';
                loadAssets();
            } else if (data.error) {
                clearInterval(interval);
                alert('Error: ' + data.error);
            }
        } catch(e) { console.error(e); }
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
            document.getElementById('titleDisplay').innerText = data.title || 'Your Video Title';
            document.getElementById('descDisplay').innerText = data.description || 'Description will appear here.';
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

# ----------------------------------------------------------------------
# Video Generation Engine (using gTTS)
# ----------------------------------------------------------------------

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
        jobs[self.job_id]['status'] = message
        jobs[self.job_id]['progress'] = progress

    async def fetch_wikipedia_content(self) -> str:
        self.update_status("Researching topic...", 5)
        try:
            import wikipedia
            search = wikipedia.search(self.theme, results=2)
            if not search:
                return f"Explore the fascinating world of {self.theme}. This complete guide covers everything you need to know."
            page = wikipedia.page(search[0])
            summary = page.summary[:1500]
            return f"{summary} In this video we'll dive deep into {self.theme} and uncover amazing facts and insights."
        except:
            return f"{self.theme} is an incredible topic. From basics to advanced concepts, this 8-minute guide will take you on a journey of discovery."

    async def generate_tts(self, text: str, output_path: Path) -> float:
        """Generate speech using gTTS (Google Text-to-Speech)"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._synth_gtts, text, output_path)
        audio = AudioFileClip(str(output_path))
        dur = audio.duration
        audio.close()
        return dur

    def _synth_gtts(self, text: str, output_path: Path):
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(output_path))

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
                                img_url = data['results'][0]['urls']['regular']
                                async with session.get(img_url) as img_resp:
                                    img_data = await img_resp.read()
                                    with open(img_path, 'wb') as f:
                                        f.write(img_data)
                                    return img_path
            except:
                pass
        # Generate artistic placeholder
        img = Image.new('RGB', (1920, 1080), color=(25, 25, 45))
        draw = ImageDraw.Draw(img)
        for i in range(1080):
            r = 25 + int(i/1080 * 60)
            g = 25 + int(i/1080 * 40)
            b = 45 + int(i/1080 * 80)
            draw.line([(0,i), (1920,i)], fill=(r,g,b))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 70)
        except:
            font = ImageFont.load_default()
        draw.text((960,540), self.theme.upper(), fill=(255,215,0), anchor="mm", font=font)
        draw.text((960,650), query[:60], fill=(200,200,200), anchor="mm", font=font)
        img.save(img_path)
        return img_path

    async def create_video_clip(self, image_path: Path, audio_path: Path):
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration
        clip = ImageClip(str(image_path)).resize(height=1080).set_duration(duration)
        # Ken Burns effect
        def make_frame(t):
            zoom = 1 + (t / duration) * 0.08
            new_w = clip.w / zoom
            new_h = clip.h / zoom
            x_center = (clip.w - new_w) * (t / duration) * 0.2
            y_center = (clip.h - new_h) * (t / duration) * 0.1
            return clip.crop(x1=x_center, y1=y_center, width=new_w, height=new_h).resize((1920,1080)).get_frame(t)
        final_clip = clip.fl(make_frame).set_audio(audio)
        return final_clip.fx(vfx.fadein, 0.5).fx(vfx.fadeout, 0.5)

    async def generate(self):
        try:
            content = await self.fetch_wikipedia_content()
            sentences = re.split(r'(?<=[.!?])\s+', content)
            # Build segments of ~15-20 sec each (~100 words)
            segments = []
            current = []
            word_count = 0
            for sent in sentences[:80]:
                words = len(sent.split())
                if word_count + words > 100:
                    if current:
                        segments.append(' '.join(current))
                    current = [sent]
                    word_count = words
                else:
                    current.append(sent)
                    word_count += words
            if current:
                segments.append(' '.join(current))
            while len(segments) < 20:
                segments.append(f"Let's continue exploring {self.theme}. Amazing insights await.")
            # Generate clips
            video_clips = []
            total_segs = len(segments)
            for i, seg_text in enumerate(segments[:35]):
                prog = 20 + int((i/total_segs)*70)
                self.update_status(f"Creating scene {i+1}/{min(35,total_segs)}", prog)
                audio_path = self.audio_dir / f"seg_{i}.mp3"
                await self.generate_tts(seg_text, audio_path)
                kw = ' '.join([w for w in seg_text.split()[:5] if len(w)>3]) or self.theme
                img_path = await self.fetch_image(kw, i)
                clip = await self.create_video_clip(img_path, audio_path)
                video_clips.append(clip)
            # Concatenate
            self.update_status("Editing video with transitions...", 85)
            final = concatenate_videoclips(video_clips, method="compose")
            # Add simple background music (optional)
            try:
                duration = final.duration
                fps = 44100
                t = np.linspace(0, duration, int(fps*duration))
                music = 0.15 * np.sin(2*np.pi*261.63*t) * np.exp(-t/40)
                music += 0.1 * np.sin(2*np.pi*329.63*t) * np.exp(-t/40)
                import soundfile as sf
                music_path = self.job_dir / "bg_music.wav"
                sf.write(music_path, music, fps)
                from moviepy.editor import AudioFileClip as AF
                bg = AF(str(music_path)).volumex(0.2)
                final = final.set_audio(bg)
            except:
                pass
            out_video = self.job_dir / "final_8min.mp4"
            final.write_videofile(str(out_video), fps=24, codec='libx264', audio_codec='aac', threads=2, preset='medium')
            # Create shorts
            self.update_status("Creating TikTok shorts...", 90)
            await self.create_shorts(out_video)
            # Thumbnail
            thumb_path = await self.generate_thumbnail()
            title, desc = await self.generate_metadata()
            jobs[self.job_id].update({
                'completed': True,
                'video_path': str(out_video),
                'short1_path': str(self.job_dir / "short1.mp4"),
                'short2_path': str(self.job_dir / "short2.mp4"),
                'thumbnail_path': str(thumb_path),
                'title': title,
                'description': desc
            })
            self.update_status("Complete! Ready to download.", 100)
        except Exception as e:
            jobs[self.job_id]['status'] = f"Error: {str(e)}"
            jobs[self.job_id]['error'] = str(e)

    async def create_shorts(self, video_path: Path):
        from moviepy.video.io.VideoFileClip import VideoFileClip
        video = VideoFileClip(str(video_path))
        dur = video.duration
        clip1 = video.subclip(0, min(60, dur))
        clip1.write_videofile(str(self.job_dir / "short1.mp4"), fps=24)
        start = max(0, (dur - 60) // 2) if dur > 120 else max(0, dur-60)
        clip2 = video.subclip(start, min(start+60, dur))
        clip2.write_videofile(str(self.job_dir / "short2.mp4"), fps=24)
        video.close()

    async def generate_thumbnail(self) -> Path:
        img = Image.new('RGB', (1920,1080), color=(15,25,45))
        draw = ImageDraw.Draw(img)
        for i in range(1080):
            r = 15 + int(i/1080*60)
            g = 25 + int(i/1080*40)
            b = 45 + int(i/1080*80)
            draw.line([(0,i),(1920,i)], fill=(r,g,b))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 55)
        except:
            font = ImageFont.load_default()
            font_small = font
        text = self.theme.upper()
        for dx,dy in [(-3,-3),(-3,3),(3,-3),(3,3)]:
            draw.text((960+dx,400+dy), text, fill=(0,0,0), anchor="mm", font=font)
        draw.text((960,400), text, fill=(255,215,0), anchor="mm", font=font)
        draw.text((960,580), "Full Documentary", fill=(255,255,255), anchor="mm", font=font_small)
        draw.text((960,680), "8 Minutes • HD", fill=(200,200,200), anchor="mm", font=font_small)
        thumb_path = self.job_dir / "thumbnail.jpg"
        img.save(thumb_path)
        return thumb_path

    async def generate_metadata(self) -> Tuple[str, str]:
        title = f"{self.theme.upper()} • Complete Guide (8-Min Documentary)"
        desc = f"""🎬 {self.theme} - Full Documentary

Discover everything about {self.theme} in this engaging 8-minute video. 

📌 What you'll learn:
- Core concepts explained
- Fascinating facts and insights
- Real-world applications
- Future developments

✨ Features:
- Cinematic visuals
- Professional narration
- Background music
- High retention editing

👍 Like & Subscribe for more deep dives!

#documentary #{self.theme.replace(' ', '')} #{self.theme}Guide
"""
        return title, desc

# ----------------------------------------------------------------------
# FastAPI Routes
# ----------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse(content=LANDING_PAGE_HTML)

@app.post("/generate")
async def start_gen(theme: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'Initializing...',
        'progress': 0,
        'completed': False,
        'theme': theme,
        'created_at': datetime.now().isoformat()
    }
    generator = VideoGenerator(theme, job_id)
    background_tasks.add_task(generator.generate)
    return JSONResponse({'job_id': job_id})

@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return JSONResponse(jobs[job_id])

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def result_page(job_id: str):
    if job_id not in jobs:
        return HTMLResponse("Job not found", status_code=404)
    html = RESULT_PAGE_HTML.replace("{{ job_id }}", job_id)
    return HTMLResponse(content=html)

@app.get("/download/{job_id}/{file_type}")
async def download(job_id: str, file_type: str):
    if job_id not in jobs:
        raise HTTPException(404)
    job = jobs[job_id]
    if not job.get('completed'):
        raise HTTPException(400, detail="Video not ready")
    mapping = {
        'video': job.get('video_path'),
        'short1': job.get('short1_path'),
        'short2': job.get('short2_path'),
        'thumbnail': job.get('thumbnail_path')
    }
    path = mapping.get(file_type)
    if not path or not Path(path).exists():
        raise HTTPException(404)
    return FileResponse(path, filename=Path(path).name)

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   🎬 AI Video Generator - Fully Self-Contained          ║
    ║   UI + Backend in one file                              ║
    ║   ▶  Running on http://localhost:8000                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
