"""
LSP Annotator – backend FastAPI
================================
Ejecutar: uv run python main.py
"""

import asyncio
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import threading
import time
import unicodedata
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

BASE_DIR        = Path(__file__).parent
VIDEOS_DIR      = BASE_DIR / "videos"
RAW_VIDEOS_DIR  = BASE_DIR / "raw_videos"
CLIPS_DIR       = BASE_DIR / "clips"
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

# ── Audio detection (Whisper API) ─────────────────────────────────────────────
AUDIO_PATRONES: dict[str, str] = {
    r"\bpara\s+qu[ée]\b":     "PARA QUÉ",   # antes de "por qué" para evitar solapamiento
    r"\bpor\s+qu[ée]\b":      "POR QUÉ",
    r"\bqu[ée]\b":             "QUÉ",
    r"\bc[oó]mo\b":            "CÓMO",
    r"\bcu[aá]ndo\b":          "CUÁNDO",
    r"\bd[oó]nde\b":           "DÓNDE",
    r"\bqui[ée]n(es)?\b":      "QUIÉN(ES)",
}

AUDIO_LABEL_MAP: dict[str, str] = {
    "QUÉ":       "¿Qué?",
    "CÓMO":      "¿Cómo?",
    "CUÁNDO":    "¿Cuándo?",
    "DÓNDE":     "¿Dónde?",
    "QUIÉN(ES)": "¿Quién?",
    "POR QUÉ":   "¿Por qué?",
    "PARA QUÉ":  "¿Para qué?",
}

AUDIO_LIMITE_API_BYTES   = 25 * 1024 * 1024   # 25 MB
AUDIO_DURACION_FRAG      = 600                  # 10 min por fragmento
AUDIO_TEMP_DIR           = BASE_DIR / "audio_temp"
AUDIO_TIMESTAMP_OFFSET   = 1    # Whisper tiende a iniciar segmentos ~1-2s antes — se corrige aquí
AUDIO_PAD_FIN            = 0.7    # segundos adicionales al final del segmento
AUDIO_MIN_DURATION       = 2.0    # duración mínima de un marcador en segundos

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
RAW_VIDEOS_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)
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
    seen: set[str] = set()
    videos = []
    for ext in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
        for f in VIDEOS_DIR.rglob(ext):
            rel      = f.relative_to(VIDEOS_DIR)
            video_id = str(rel.with_suffix(""))
            if video_id in seen:
                continue
            seen.add(video_id)
            videos.append({
                "id":       video_id,          # ruta relativa sin extensión, p.ej. "a/b/video1"
                "filename": f.name,            # solo el nombre, p.ej. "video1.mp4"
                "path":     str(rel),          # ruta relativa con extensión, p.ej. "a/b/video1.mp4"
                "duration": None,
            })
    videos.sort(key=lambda v: v["path"])
    return JSONResponse(content=videos)


# ---------------------------------------------------------------------------
# GET /videos/{filename}  – con soporte de Range requests
# ---------------------------------------------------------------------------

@app.get("/videos/{path:path}")
async def serve_video(path: str, request: Request):
    video_file = VIDEOS_DIR / path
    if not video_file.exists() or not video_file.is_file():
        raise HTTPException(status_code=404, detail="Video no encontrado")

    file_size    = video_file.stat().st_size
    filename     = Path(path).name
    content_type = "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4"
    range_header = request.headers.get("range")

    if not range_header:
        # Sin Range: FileResponse envía Content-Length correcto y soporta seeking.
        # StreamingResponse con generator usa Transfer-Encoding: chunked, lo que
        # impide que el browser detecte la pista de audio.
        return FileResponse(video_file, media_type=content_type,
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
    with open(video_file, "rb") as f:
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

@app.get("/annotations/{video_id:path}")
async def get_annotations(video_id: str):
    ann_file = ANNOTATIONS_DIR / f"{video_id}.json"
    if not ann_file.exists():
        return JSONResponse(content={"video_id": video_id, "annotations": []})
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# POST /annotations/{video_id}
# ---------------------------------------------------------------------------

@app.post("/annotations/{video_id:path}")
async def save_annotations(video_id: str, request: Request):
    body = await request.json()
    ann_file = ANNOTATIONS_DIR / f"{video_id}.json"
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    ann_file.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse(content={"ok": True})


# ---------------------------------------------------------------------------
# GET /clips/{video_id}  – descarga un segmento del video usando ffmpeg
# ---------------------------------------------------------------------------

def _normalizar_para_nombre(texto: str, max_len: int = 50) -> str:
    """Convierte texto libre en un slug seguro para nombres de archivo."""
    texto = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^\w\s]", "", texto).strip()
    texto = re.sub(r"\s+", "_", texto)
    return texto[:max_len].strip("_")


@app.get("/clips/{video_id:path}")
async def download_clip(
    video_id: str,
    start:   float = Query(..., ge=0, description="Tiempo de inicio en segundos"),
    end:     float = Query(..., gt=0, description="Tiempo de fin en segundos"),
    label:   str   = Query("", description="Etiqueta de la anotación"),
    oracion: str   = Query("", description="Frase de referencia (opcional)"),
):
    if end <= start:
        raise HTTPException(status_code=400, detail="'end' debe ser mayor que 'start'")

    # Buscar el archivo de video (video_id es la ruta relativa sin extensión)
    video_path: Path | None = None
    for ext in (".mp4", ".mov", ".MP4", ".MOV"):
        candidate = VIDEOS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    if shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=501,
            detail="ffmpeg no está instalado. Instálalo para poder descargar clips.",
        )

    # Construir nombre y ruta del clip siguiendo la estructura de carpetas
    video_stem = Path(video_id).name

    # Normalizar etiqueta y oración para el nombre de archivo
    label_norm  = _normalizar_para_nombre(label).upper() or "CLIP"
    oracion_norm = _normalizar_para_nombre(oracion, 30)

    rel_subdir   = Path(video_id).parent
    clips_subdir = CLIPS_DIR / rel_subdir
    clips_subdir.mkdir(parents=True, exist_ok=True)

    # Evitar sobreescritura: primera vez sin número, luego _2, _3, …
    suffix    = f"_{oracion_norm}" if oracion_norm else ""
    base_name = f"{video_stem}_clip_{label_norm}{suffix}"
    clip_name   = f"{base_name}.mp4"
    clip_output = clips_subdir / clip_name
    counter     = 2
    while clip_output.exists():
        clip_name   = f"{base_name}_{counter}.mp4"
        clip_output = clips_subdir / clip_name
        counter    += 1

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),
                "-c", "copy",
                str(clip_output),
            ],
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout al extraer el clip")

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail="ffmpeg falló al extraer el clip")

    return JSONResponse(content={"ok": True, "clip": clip_name})


# ---------------------------------------------------------------------------
# Conversión MTS → MP4  (30 fps, H.264 + AAC)
# ---------------------------------------------------------------------------

# Almacena jobs en memoria: {job_id: {status, progress, error}}
_mts_jobs: dict[str, dict] = {}
_MTS_EXTS = {".mts", ".MTS"}


@app.get("/mts-files")
async def list_mts_files():
    """Lista los archivos .mts presentes en raw_videos/ (incluyendo subcarpetas)."""
    files = []
    for f in RAW_VIDEOS_DIR.rglob("*"):
        if f.suffix.lower() == ".mts":
            rel  = f.relative_to(RAW_VIDEOS_DIR)
            stem = str(rel.with_suffix(""))
            files.append({
                "stem":      stem,
                "filename":  f.name,
                "size_mb":   round(f.stat().st_size / 1_000_000, 1),
                "converted": (VIDEOS_DIR / f"{stem}.mp4").exists(),
            })
    files.sort(key=lambda x: x["stem"])
    return JSONResponse(content=files)


@app.post("/convert-mts/{stem:path}")
async def start_convert_mts(stem: str):
    """Inicia la conversión en background de raw_videos/stem.mts → videos/stem.mp4 a 30 fps, máximo 480p."""
    mts_path: Path | None = None
    for ext in _MTS_EXTS:
        candidate = RAW_VIDEOS_DIR / f"{stem}{ext}"
        if candidate.exists():
            mts_path = candidate
            break
    if mts_path is None:
        raise HTTPException(status_code=404, detail="Archivo .mts no encontrado en raw_videos/")

    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=501, detail="ffmpeg no está instalado")

    job_id = str(uuid.uuid4())
    _mts_jobs[job_id] = {"status": "running", "progress": 0, "error": None}

    def _run() -> None:
        output = VIDEOS_DIR / f"{stem}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
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

        # Reencoding a libx264, máximo 480p de altura, preservando relación de aspecto
        cmd_encode = [
            "ffmpeg", "-y",
            "-i", str(mts_path),
            "-r", "30",
            "-vf", "scale=-2:min(480\\,ih)",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac",
            str(output),
        ]

        try:
            _mts_jobs[job_id]["mode"] = "encode"
            rc = _run_cmd(cmd_encode)
            if rc == 0:
                # Eliminar el archivo MTS original tras conversión exitosa
                try:
                    mts_path.unlink()
                    # Borrar carpeta padre si quedó vacía
                    parent = mts_path.parent
                    while parent != RAW_VIDEOS_DIR:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                except Exception:
                    pass
                _mts_jobs[job_id].update({"status": "done", "progress": 100, "error": None})
            else:
                _mts_jobs[job_id].update({"status": "error", "error": "ffmpeg falló al convertir"})
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


@app.post("/analyze/{video_id:path}")
async def analyze_video(video_id: str):
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no cargado. Asegurate de que exista: {MODEL_PATH}",
        )

    video_path: Path | None = None
    for ext in (".mp4", ".mov", ".MP4", ".MOV"):
        candidate = VIDEOS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            video_path = candidate
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
# Detección de palabras interrogativas por audio (Whisper API)
# ---------------------------------------------------------------------------

def _detectar_audio(video_path: Path, job: dict) -> list[dict]:
    """
    Extrae el audio del video, lo transcribe con Whisper y busca palabras
    interrogativas.  Actualiza `job` con progreso (0-100).
    Devuelve lista de {id, start, end, label, source, confidence}.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no está configurada. "
            "Defínela en el entorno antes de iniciar el servidor."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    AUDIO_TEMP_DIR.mkdir(exist_ok=True)
    audio_path = AUDIO_TEMP_DIR / f"{uuid.uuid4().hex}.wav"

    # ── Fase 1: extraer audio (0 → 15%) ──────────────────────────────────────
    job.update({"phase": "Extrayendo audio…", "progress": 5})
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        capture_output=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al extraer audio: {result.stderr.decode()[:300]}")

    job.update({"progress": 15})

    # ── Fase 2: dividir si supera 25 MB (15 → 30%) ───────────────────────────
    tamano = audio_path.stat().st_size
    fragmentos: list[tuple[Path, float]] = []  # (ruta, offset_segundos)

    if tamano <= AUDIO_LIMITE_API_BYTES:
        fragmentos = [(audio_path, 0.0)]
    else:
        job.update({"phase": "Dividiendo audio…", "progress": 20})
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        duracion = float(probe.stdout.strip() or "0")
        n_frags  = math.ceil(duracion / AUDIO_DURACION_FRAG)
        frag_dir = AUDIO_TEMP_DIR / audio_path.stem
        frag_dir.mkdir(exist_ok=True)
        for i in range(n_frags):
            offset    = i * AUDIO_DURACION_FRAG
            frag_path = frag_dir / f"frag_{i:03d}.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ss", str(offset), "-t", str(AUDIO_DURACION_FRAG),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(frag_path),
                ],
                capture_output=True, timeout=300,
            )
            fragmentos.append((frag_path, float(offset)))
        audio_path.unlink(missing_ok=True)

    job.update({"progress": 30})

    # ── Fase 3: transcribir con Whisper (30 → 90%) ───────────────────────────
    todos_segmentos: list[dict] = []
    n_frags = len(fragmentos)

    for idx, (frag_path, offset) in enumerate(fragmentos):
        job.update({
            "phase":    f"Transcribiendo… ({idx + 1}/{n_frags})",
            "progress": 30 + int((idx / n_frags) * 60),
        })
        with open(frag_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="es",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        for seg in (resp.model_dump().get("segments") or []):
            seg["start"] = (seg.get("start") or 0) + offset
            seg["end"]   = (seg.get("end")   or 0) + offset
            todos_segmentos.append(seg)

    # ── Fase 4: detectar patrones y construir anotaciones (90 → 100%) ────────
    job.update({"phase": "Detectando palabras interrogativas…", "progress": 90})

    anotaciones: list[dict] = []
    for seg in todos_segmentos:
        texto = (seg.get("text") or "").strip()
        if not texto:
            continue
        texto_lower = texto.lower()
        encontradas: list[str] = []
        for patron, etiqueta_raw in AUDIO_PATRONES.items():
            if re.search(patron, texto_lower, flags=re.IGNORECASE):
                label = AUDIO_LABEL_MAP.get(etiqueta_raw, etiqueta_raw)
                if label not in encontradas:
                    encontradas.append(label)
        for label in encontradas:
            t_ini = round(max(0.0, float(seg["start"]) + AUDIO_TIMESTAMP_OFFSET), 2)
            t_fin = round(float(seg["end"]) + AUDIO_TIMESTAMP_OFFSET + AUDIO_PAD_FIN, 2)
            if t_fin - t_ini < AUDIO_MIN_DURATION:
                t_fin = round(t_ini + AUDIO_MIN_DURATION, 2)
            anotaciones.append({
                "id":     str(uuid.uuid4()),
                "time":   t_ini,
                "end":    t_fin,
                "label":  label,
                "oracion": texto,
            })

    # ── Limpieza ──────────────────────────────────────────────────────────────
    try:
        audio_path.unlink(missing_ok=True)
        if n_frags > 1:
            shutil.rmtree(frag_dir, ignore_errors=True)
    except Exception:
        pass

    return anotaciones


_audio_jobs: dict[str, dict] = {}


@app.post("/analyze-audio/{video_id:path}")
async def analyze_audio(video_id: str):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY no está configurada. Defínela en el entorno antes de iniciar el servidor.",
        )

    video_path: Path | None = None
    for ext in (".mp4", ".mov", ".MP4", ".MOV"):
        candidate = VIDEOS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=501, detail="ffmpeg no está instalado")

    job_id = str(uuid.uuid4())
    job    = {"status": "running", "progress": 0, "phase": "Iniciando…",
              "annotations": None, "error": None}
    _audio_jobs[job_id] = job

    def _run():
        try:
            anns = _detectar_audio(video_path, job)
            job.update({"status": "done", "progress": 100,
                        "phase": f"{len(anns)} palabras detectadas", "annotations": anns})
        except Exception as exc:
            job.update({"status": "error", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse(content={"job_id": job_id})


@app.get("/analyze-audio-status/{job_id}")
async def analyze_audio_status(job_id: str):
    job = _audio_jobs.get(job_id)
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
