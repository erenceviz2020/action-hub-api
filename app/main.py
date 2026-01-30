import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI

app = FastAPI(title="Action Hub API", version="0.2.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MVP için pratik limit: 25 MB (yaklaşık 10 dk çoğu konuşma kaydı için yeterli olur)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "audio/mpeg",   # mp3
    "audio/mp4",    # m4a
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_hint: str | None = None,  # "tr" veya "en" verebilirsin, boş bırakılabilir
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
        raise HTTPException(
            status_code=413,
            detail=f"file too large (max {MAX_UPLOAD_BYTES} bytes)",
        )

    # OpenAI SDK bu endpoint için file-like object bekliyor.
    # UploadFile stream'ini başa sarıp kullanacağız.
    await file.seek(0)

    try:
        kwargs = {}
        if language_hint:
            # ISO-639-1 ("tr", "en")
            kwargs["language"] = language_hint

        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=file.file,
            **kwargs,
        )

        # SDK genelde resp.text döndürür
        text = getattr(resp, "text", None)
        if not text:
            # bazı durumlarda dict dönebilir
            text = resp.get("text") if isinstance(resp, dict) else None

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(data),
            "language_hint": language_hint,
            "text": text or "",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")