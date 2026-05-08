# AI Video Generator

Generates complete 8‑min videos, TikTok shorts, thumbnail, and metadata from a single theme.

## Deploy on Render

1. Click "New Web Service" and connect your GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Optionally add an `UNSPLASH_ACCESS_KEY` environment variable for better images.
