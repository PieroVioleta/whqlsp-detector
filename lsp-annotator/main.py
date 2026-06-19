"""
LSP Annotator – backend FastAPI
================================
Ejecutar: uv run python main.py
"""

import asyncio
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

BASE_DIR        = Path(__file__).parent
VIDEOS_DIR      = BASE_DIR / "videos"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
FRONTEND_DIR    = BASE_DIR / "frontend"

# Ruta al modelo — busca primero junto al anotador, luego en notebooks/models/
_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "svm_lsp.pkl",
    BASE_DIR.parent / "notebooks" / "models" / "svm_lsp.pkl",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[-1])

# ── Parámetros de inferencia ──────────────────────────────────────────────────
INFER_TARGET_FPS  = 15     # FPS a muestrear del video (el modelo normaliza igual)
INFER_STRIDE      = 5      # avance de ventana deslizante (en frames muestreados)
INFER_UMBRAL_CONF = 0.80   # confianza mínima para reportar una seña
INFER_UMBRAL_NADA = 0.50   # prob. de NADA a partir de la cual se silencia
INFER_MIN_DUR     = 0.5    # duración mínima en segundos para reportar ocurrencia

# Mapeo de etiquetas internas → etiquetas del anotador
LABEL_MAP: dict[str, str] = {
    'QUE':     '¿Qué?',
    'QUIEN':   '¿Quién?',
    'CUANDO':  '¿Cuándo?',
    'DONDE':   '¿Dónde?',
    'COMO':    '¿Cómo?',
    'PORQUE':  '¿Por qué?',
    'POR_QUE': '¿Por qué?',   # variante con guión bajo (ej: clase POR_QUE1)
    'CUANTO':  '¿Cuánto?',
    'CUANTOS': '¿Cuánto?',
}

# ── Estado global del modelo (cargado al arrancar) ────────────────────────────
_bundle: dict | None = None
_hands:  Any         = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle, _hands
    # Cargar modelo
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
        print(f"[modelo] Cargado: {MODEL_PATH.name}")
        print(f"[modelo] Clases: {_bundle['clases']}  |  CV accuracy: {_bundle['cv_accuracy']:.3f}")
    else:
        print(f"[modelo] ADVERTENCIA: no se encontró el modelo en {MODEL_PATH}")
        print(f"[modelo] El botón 'Analizar' devolverá error hasta que exista el .pkl")

    # Inicializar MediaPipe (una sola instancia compartida)
    _hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,            # modelo liviano: ~30% más rápido, suficiente para señas
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[mediapipe] Inicializado (model_complexity=0)")
    yield
    # Cleanup
    if _hands:
        _hands.close()


VIDEOS_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="LSP Annotator", lifespan=lifespan)


# ── Funciones de inferencia ───────────────────────────────────────────────────

def _limpiar_etiqueta(raw: str) -> str:
    """'QUE1' → 'QUE', 'QUIEN1' → 'QUIEN'."""
    return raw.rstrip("0123456789")


def _extraer_landmarks(video_path: Path) -> tuple[np.ndarray, float]:
    """Extrae landmarks muestreando a INFER_TARGET_FPS. Devuelve (T,2,21,3) y fps_efectivo."""
    cap = cv2.VideoCapture(str(video_path))
    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step     = max(1, round(fps_orig / INFER_TARGET_FPS))
    fps_ef   = fps_orig / step

    landmarks, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res      = _hands.process(rgb)
            frame_lm = np.zeros((2, 21, 3), dtype=np.float32)
            if res.multi_hand_landmarks:
                for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    hi = 0 if handedness.classification[0].label == "Right" else 1
                    for j, lm in enumerate(hand_lm.landmark):
                        frame_lm[hi, j] = [lm.x, lm.y, lm.z]
            landmarks.append(frame_lm)
        idx += 1
    cap.release()
    return np.array(landmarks, dtype=np.float32), fps_ef


def _normalizar(seq: np.ndarray, n: int) -> np.ndarray:
    T = len(seq)
    if T == n:
        return seq
    idx = np.linspace(0, T - 1, n)
    out = np.zeros((n, 2, 21, 3), dtype=np.float32)
    for i, x in enumerate(idx):
        i0 = int(x); i1 = min(i0 + 1, T - 1); a = x - i0
        out[i] = seq[i0] * (1 - a) + seq[i1] * a
    return out


def _predecir_ventana(probs: np.ndarray, clases: list) -> tuple[str, float] | None:
    """Aplica lógica de umbrales duales. Devuelve (etiqueta_limpia, conf) o None."""
    if "NADA" in clases:
        if probs[list(clases).index("NADA")] >= INFER_UMBRAL_NADA:
            return None
    idx_max   = int(probs.argmax())
    confianza = float(probs[idx_max])
    etiqueta  = _limpiar_etiqueta(clases[idx_max])
    if etiqueta == "NADA" or confianza < INFER_UMBRAL_CONF:
        return None
    return etiqueta, confianza


def _detectar_senas(video_path: Path, job: dict) -> list[dict]:
    """
    Pipeline completo: extracción → ventana deslizante → agrupación.
    Actualiza `job` con progreso (0-100) y fase actual.
    Devuelve lista de {start, end, label, confidence}.
    """
    if _bundle is None:
        raise RuntimeError("Modelo no cargado")

    n_frames = _bundle["n_frames"]
    clases   = list(_bundle["label_encoder"].classes_)

    # ── Fase 1: extracción de landmarks (0 → 70%) ────────────────────────────
    cap      = cv2.VideoCapture(str(video_path))
    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step     = max(1, round(fps_orig / INFER_TARGET_FPS))
    fps_ef   = fps_orig / step
    n_sample = (total_f + step - 1) // step

    job.update({"phase": "Extrayendo landmarks…", "progress": 0})

    landmarks, frame_idx, sampled = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res      = _hands.process(rgb)
            frame_lm = np.zeros((2, 21, 3), dtype=np.float32)
            if res.multi_hand_landmarks:
                for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    hi = 0 if handedness.classification[0].label == "Right" else 1
                    for j, lm in enumerate(hand_lm.landmark):
                        frame_lm[hi, j] = [lm.x, lm.y, lm.z]
            landmarks.append(frame_lm)
            sampled += 1
            # Actualizar progreso cada 30 frames muestreados
            if sampled % 30 == 0:
                job["progress"] = min(69, int(sampled / n_sample * 70))
        frame_idx += 1
    cap.release()

    landmarks = np.array(landmarks, dtype=np.float32)
    T         = len(landmarks)

    # ── Fase 2: ventana deslizante (70 → 95%) ────────────────────────────────
    job.update({"phase": "Clasificando señas…", "progress": 70})

    ventanas    = list(range(0, T - n_frames + 1, INFER_STRIDE))
    n_ventanas  = max(len(ventanas), 1)
    detecciones = []

    for vi, inicio in enumerate(ventanas):
        ventana = _normalizar(landmarks[inicio : inicio + n_frames], n_frames)
        probs   = _bundle["pipeline"].predict_proba(ventana.flatten().reshape(1, -1))[0]
        pred    = _predecir_ventana(probs, clases)
        if pred:
            etiqueta, conf = pred
            detecciones.append({
                "frame_inicio": inicio,
                "frame_fin":    inicio + n_frames,
                "t_inicio":     inicio / fps_ef,
                "t_fin":        (inicio + n_frames) / fps_ef,
                "clase":        etiqueta,
                "confianza":    conf,
            })
        if vi % 20 == 0:
            job["progress"] = 70 + min(24, int(vi / n_ventanas * 25))

    # ── Fase 3: agrupación y formato (95 → 100%) ─────────────────────────────
    job.update({"phase": "Agrupando ocurrencias…", "progress": 95})

    ocurrencias = []
    if detecciones:
        actual = {**detecciones[0], "confianzas": [detecciones[0]["confianza"]]}
        for d in detecciones[1:]:
            mismo   = d["clase"] == actual["clase"]
            sin_gap = d["frame_inicio"] <= actual["frame_fin"] + n_frames
            if mismo and sin_gap:
                actual["frame_fin"] = max(actual["frame_fin"], d["frame_fin"])
                actual["t_fin"]     = actual["frame_fin"] / fps_ef
                actual["confianzas"].append(d["confianza"])
            else:
                if actual["t_fin"] - actual["t_inicio"] >= INFER_MIN_DUR:
                    ocurrencias.append(actual)
                actual = {**d, "confianzas": [d["confianza"]]}
        if actual["t_fin"] - actual["t_inicio"] >= INFER_MIN_DUR:
            ocurrencias.append(actual)

    anotaciones = []
    for oc in ocurrencias:
        label = LABEL_MAP.get(oc["clase"], oc["clase"])
        anotaciones.append({
            "id":         str(uuid.uuid4()),
            "start":      round(oc["t_inicio"], 2),
            "end":        round(oc["t_fin"], 2),
            "label":      label,
            "source":     "model",
            "confidence": round(float(np.mean(oc["confianzas"])), 3),
        })
    return anotaciones

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

    file_size    = path.stat().st_size
    content_type = "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4"
    range_header = request.headers.get("range")

    if not range_header:
        # Sin Range: FileResponse envía Content-Length correcto y soporta seeking.
        # StreamingResponse con generator usa Transfer-Encoding: chunked, lo que
        # impide que el browser detecte la pista de audio.
        return FileResponse(path, media_type=content_type,
                            headers={"Accept-Ranges": "bytes"})

    # Con Range: el browser pide chunks pequeños (metadata, buffer de reproducción).
    # Leer en memoria es seguro — los chunks típicos son < 2 MB.
    m = re.match(r"bytes=(\d*)-(\d*)", range_header)
    start = 0
    end   = file_size - 1
    if m:
        s, e = m.group(1), m.group(2)
        start = int(s) if s else 0
        end   = int(e) if e else file_size - 1
        end   = min(end, file_size - 1)

    chunk_size = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    headers = {
        "Content-Range":  f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges":  "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type":   content_type,
    }
    return Response(content=data, status_code=206, headers=headers)


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
# GET /clips/{video_id}  – descarga un segmento del video usando ffmpeg
# ---------------------------------------------------------------------------

@app.get("/clips/{video_id}")
async def download_clip(
    video_id: str,
    start: float = Query(..., ge=0, description="Tiempo de inicio en segundos"),
    end: float   = Query(..., gt=0, description="Tiempo de fin en segundos"),
):
    if end <= start:
        raise HTTPException(status_code=400, detail="'end' debe ser mayor que 'start'")

    # Buscar el archivo de video
    video_path: Path | None = None
    for f in VIDEOS_DIR.iterdir():
        if f.stem == video_id and f.suffix.lower() in (".mp4", ".mov"):
            video_path = f
            break
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    if shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=501,
            detail="ffmpeg no está instalado. Instálalo para poder descargar clips.",
        )

    suffix = video_path.suffix.lower()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),
                "-c", "copy",
                tmp_path,
            ],
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        raise HTTPException(status_code=504, detail="Timeout al extraer el clip")

    if result.returncode != 0:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail="ffmpeg falló al extraer el clip")

    media_type = "video/quicktime" if suffix == ".mov" else "video/mp4"
    clip_name  = f"{video_id}_{start:.2f}s-{end:.2f}s{suffix}"

    def _cleanup():
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return FileResponse(
        tmp_path,
        media_type=media_type,
        filename=clip_name,
        background=BackgroundTask(_cleanup),
    )


# ---------------------------------------------------------------------------
# Conversión MTS → MP4  (30 fps, H.264 + AAC)
# ---------------------------------------------------------------------------

# Almacena jobs en memoria: {job_id: {status, progress, error}}
_mts_jobs: dict[str, dict] = {}
_MTS_EXTS = {".mts", ".MTS"}


@app.get("/mts-files")
async def list_mts_files():
    """Lista los archivos .mts presentes en la carpeta videos/."""
    files = []
    for f in VIDEOS_DIR.iterdir():
        if f.suffix in _MTS_EXTS:
            files.append({
                "stem":      f.stem,
                "filename":  f.name,
                "size_mb":   round(f.stat().st_size / 1_000_000, 1),
                "converted": (VIDEOS_DIR / f"{f.stem}.mp4").exists(),
            })
    files.sort(key=lambda x: x["filename"])
    return JSONResponse(content=files)


@app.post("/convert-mts/{stem}")
async def start_convert_mts(stem: str):
    """Inicia la conversión en background de stem.mts → stem.mp4 a 30 fps."""
    mts_path: Path | None = None
    for ext in _MTS_EXTS:
        candidate = VIDEOS_DIR / f"{stem}{ext}"
        if candidate.exists():
            mts_path = candidate
            break
    if mts_path is None:
        raise HTTPException(status_code=404, detail="Archivo .mts no encontrado")

    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=501, detail="ffmpeg no está instalado")

    job_id = str(uuid.uuid4())
    _mts_jobs[job_id] = {"status": "running", "progress": 0, "error": None}

    def _run() -> None:
        output = VIDEOS_DIR / f"{stem}.mp4"
        total_secs = 0.0

        # Obtener duración con ffprobe (si está disponible)
        if shutil.which("ffprobe"):
            try:
                r = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(mts_path),
                    ],
                    capture_output=True, text=True, timeout=15,
                )
                total_secs = float(r.stdout.strip())
            except Exception:
                pass

        def _run_cmd(cmd: list[str]) -> int:
            """Ejecuta ffmpeg, parsea progreso desde stderr y retorna el returncode."""
            proc = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            time_re = re.compile(r"time=(\d+):(\d+):([\d.]+)")
            for line in proc.stderr:  # type: ignore[union-attr]
                m = time_re.search(line)
                if m and total_secs > 0:
                    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    elapsed = h * 3600 + mn * 60 + s
                    _mts_jobs[job_id]["progress"] = min(99, int(elapsed / total_secs * 100))
            proc.wait()
            return proc.returncode

        # Intento 1: remux sin reencoding (rápido, sin pérdida de calidad)
        cmd_copy = [
            "ffmpeg", "-y",
            "-i", str(mts_path),
            "-c", "copy",
            str(output),
        ]
        # Intento 2 (fallback): reencoding libx264 si el primer intento falla
        cmd_encode = [
            "ffmpeg", "-y",
            "-i", str(mts_path),
            "-r", "30",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac",
            str(output),
        ]

        try:
            _mts_jobs[job_id]["mode"] = "copy"
            rc = _run_cmd(cmd_copy)
            if rc != 0:
                # Fallback: reencoding
                _mts_jobs[job_id].update({"mode": "encode", "progress": 0})
                rc = _run_cmd(cmd_encode)
            if rc == 0:
                _mts_jobs[job_id].update({"status": "done", "progress": 100, "error": None})
            else:
                _mts_jobs[job_id].update({"status": "error", "error": "ffmpeg falló en ambos modos"})
        except Exception as exc:
            _mts_jobs[job_id].update({"status": "error", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse(content={"job_id": job_id})


@app.get("/convert-status/{job_id}")
async def convert_status(job_id: str):
    """Consulta el estado de un job de conversión."""
    job = _mts_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return JSONResponse(content=job)


# ---------------------------------------------------------------------------
# POST /analyze/{video_id}  – inicia job de inferencia en background
# GET  /analyze-status/{job_id} – consulta progreso
# ---------------------------------------------------------------------------

_analyze_jobs: dict[str, dict] = {}


@app.post("/analyze/{video_id}")
async def analyze_video(video_id: str):
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no cargado. Asegurate de que exista: {MODEL_PATH}",
        )

    video_path: Path | None = None
    for f in VIDEOS_DIR.iterdir():
        if f.stem == video_id and f.suffix.lower() in (".mp4", ".mov"):
            video_path = f
            break
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    job_id = str(uuid.uuid4())
    job    = {"status": "running", "progress": 0, "phase": "Iniciando…",
              "annotations": None, "error": None}
    _analyze_jobs[job_id] = job

    def _run():
        try:
            anns = _detectar_senas(video_path, job)
            job.update({"status": "done", "progress": 100,
                        "phase": f"{len(anns)} señas detectadas", "annotations": anns})
        except Exception as exc:
            job.update({"status": "error", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse(content={"job_id": job_id})


@app.get("/analyze-status/{job_id}")
async def analyze_status(job_id: str):
    job = _analyze_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return JSONResponse(content=job)


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
