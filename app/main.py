import os

from fastapi import FastAPI
from openai import OpenAI

app = FastAPI(title="Action Hub API", version="0.1.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/debug/openai")
def debug_openai():
    models = client.models.list()
    return {"ok": True, "model_count": len(models.data)}