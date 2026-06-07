"""
LSP Annotator – backend FastAPI
================================
Ejecutar: uv run python main.py
"""

import json
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
FRONTEND_DIR = BASE_DIR / "frontend"

VIDEOS_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="LSP Annotator")

# ---------------------------------------------------------------------------
# Raíz → sirve index.html
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# GET /videos
# ---------------------------------------------------------------------------

@app.get("/videos")
async def list_videos():
    videos = []
    for ext in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
        for f in VIDEOS_DIR.glob(ext):
            video_id = f.stem
            videos.append({
                "id": video_id,
                "filename": f.name,
                "duration": None,
            })
    videos.sort(key=lambda v: v["filename"])
    return JSONResponse(content=videos)


# ---------------------------------------------------------------------------
# GET /videos/{filename}  – con soporte de Range requests
# ---------------------------------------------------------------------------

@app.get("/videos/{filename}")
async def serve_video(filename: str, request: Request):
    path = VIDEOS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Video no encontrado")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    content_type = "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4"

    if range_header:
        # Parsear "bytes=start-end"
        range_value = range_header.replace("bytes=", "")
        parts = range_value.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(chunk_size)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": content_type,
        }
        return Response(content=data, status_code=206, headers=headers)

    # Sin Range: devuelve el archivo completo (videos pequeños / primera carga)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": content_type,
    }
    return FileResponse(path, headers=headers, media_type=content_type)


# ---------------------------------------------------------------------------
# GET /annotations/{video_id}
# ---------------------------------------------------------------------------

@app.get("/annotations/{video_id}")
async def get_annotations(video_id: str):
    ann_file = ANNOTATIONS_DIR / f"{video_id}.json"
    if not ann_file.exists():
        return JSONResponse(content={"video_id": video_id, "annotations": []})
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# POST /annotations/{video_id}
# ---------------------------------------------------------------------------

@app.post("/annotations/{video_id}")
async def save_annotations(video_id: str, request: Request):
    body = await request.json()
    ann_file = ANNOTATIONS_DIR / f"{video_id}.json"
    ann_file.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse(content={"ok": True})


# ---------------------------------------------------------------------------
# POST /analyze/{video_id}  – MOCK del modelo
# ---------------------------------------------------------------------------

@app.post("/analyze/{video_id}")
async def analyze_video(video_id: str):
    # TODO: reemplazar con inferencia real del modelo
    mock_annotations: list[dict[str, Any]] = [
        {"id": str(uuid.uuid4()), "start": 8.2,  "end": 10.5, "label": "¿Quién?",  "source": "model", "confidence": 0.94},
        {"id": str(uuid.uuid4()), "start": 23.1, "end": 25.0, "label": "¿Dónde?",  "source": "model", "confidence": 0.87},
        {"id": str(uuid.uuid4()), "start": 41.7, "end": 43.2, "label": "¿Qué?",    "source": "model", "confidence": 0.91},
        {"id": str(uuid.uuid4()), "start": 67.0, "end": 68.8, "label": "¿Cuándo?", "source": "model", "confidence": 0.78},
    ]
    return JSONResponse(content={"video_id": video_id, "annotations": mock_annotations})


# ---------------------------------------------------------------------------
# Arranque: abrir browser automáticamente
# ---------------------------------------------------------------------------

def _open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
