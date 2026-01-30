from fastapi import FastAPI

app = FastAPI(title="Action Hub API", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}