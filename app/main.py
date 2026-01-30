import os
import re
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Action Hub API", version="0.3.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "audio/mpeg",   # mp3
    "audio/mp4",    # m4a
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
}

YOUTUBE_URL_RE = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/")


class YouTubeIngestRequest(BaseModel):
    url: str
    language_hint: str | None = None  # "tr" / "en" / None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_hint: str | None = None,
):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported content_type: {file.content_type}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file too large (max {MAX_UPLOAD_BYTES} bytes)")

    await file.seek(0)

    try:
        kwargs = {}
        if language_hint:
            kwargs["language"] = language_hint

        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=file.file,
            **kwargs,
        )
        text = getattr(resp, "text", "") or ""
        return {
            "source": "upload",
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(data),
            "language_hint": language_hint,
            "text": text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")


@app.post("/v1/ingest/youtube")
def ingest_youtube(payload: YouTubeIngestRequest):
    url = payload.url.strip()
    if not YOUTUBE_URL_RE.match(url):
        raise HTTPException(status_code=400, detail="only youtube URLs are supported")

    # Çıktı dosyalarını temp klasörde tutuyoruz
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        outtmpl = str(tmp / "audio.%(ext)s")

        # 1) yt-dlp ile en iyi audio'yu indir
        # --no-playlist: playlist olmasın
        # -x: audio extract
        # --audio-format mp3: mp3'e dönüştür (ffmpeg ile)
       cmd = [
    "yt-dlp",
    "--js-runtimes",
    "node",
    "--no-playlist",
    "-x",
    "--audio-format",
    "mp3",
    "-o",
    outtmpl,
    url,
]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="download timed out (over 10 minutes)")

        if r.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"yt-dlp failed: {r.stderr[-800:] if r.stderr else r.stdout[-800:]}",
            )

        # İndirilen mp3'ü bul
        mp3_files = list(tmp.glob("audio*.mp3"))
        if not mp3_files:
            # bazı durumlarda ext farklı olabilir, ne varsa bul
            any_files = list(tmp.glob("audio.*"))
            raise HTTPException(status_code=500, detail=f"audio file not found. found={ [f.name for f in any_files] }")

        audio_path = mp3_files[0]

        # 2) OpenAI transcribe (file handle ile)
        try:
            with open(audio_path, "rb") as f:
                kwargs = {}
                if payload.language_hint:
                    kwargs["language"] = payload.language_hint

                resp = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                    **kwargs,
                )
                text = getattr(resp, "text", "") or ""

            return {
                "source": "youtube",
                "url": url,
                "language_hint": payload.language_hint,
                "audio_bytes": audio_path.stat().st_size,
                "text": text,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"transcription failed: {e}")