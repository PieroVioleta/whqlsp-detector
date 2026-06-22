"""
Detección de palabras interrogativas en un video, con contexto de oración,
usando la API de Whisper (OpenAI) con timestamps a nivel de SEGMENTO (oración).

Flujo:
1. Extrae el audio del video con ffmpeg.
2. Si el audio pesa más de 25MB (límite de la API), lo divide en fragmentos.
3. Transcribe cada fragmento con whisper-1 pidiendo timestamps por segmento (oración).
4. Busca las palabras interrogativas dentro del texto de cada oración y
   reporta el segmento completo (inicio-fin) donde aparece, con su contexto.

Requisitos:
    pip install openai
    ffmpeg instalado en el sistema

Uso:
    export OPENAI_API_KEY="tu-key-aqui"
    python detectar_palabras.py ruta/al/video.mp4
    python detectar_palabras.py "ruta con espacios/video.mp4"
"""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

MODELO_TRANSCRIPCION = "whisper-1"

PATRONES_CLAVE = {
    r"\bpara\s+qu[ée]\b":     "PARA QUÉ",   # antes de "por qué" para evitar solapamiento
    r"\bpor\s+qu[ée]\b":      "POR QUÉ",
    r"\bqu[ée]\b":             "QUÉ",
    r"\bc[oó]mo\b":            "CÓMO",
    r"\bcu[aá]ndo\b":          "CUÁNDO",
    r"\bd[oó]nde\b":           "DÓNDE",
    r"\bqui[ée]n(es)?\b":      "QUIÉN(ES)",
}

LIMITE_API_BYTES   = 25 * 1024 * 1024
DURACION_FRAGMENTO = 600  # 10 minutos


def extraer_audio(video_path: str, audio_path: str):
    print(f"Extrayendo audio de '{video_path}'...")
    resultado = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"Error al extraer audio con ffmpeg:\n{resultado.stderr}")
    print(f"Audio extraído: {audio_path}")


def obtener_duracion(path: str) -> float:
    resultado = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return float(resultado.stdout.strip() or "0")


def dividir_audio(audio_path: str, carpeta_salida: str) -> list:
    duracion  = obtener_duracion(audio_path)
    n_frags   = math.ceil(duracion / DURACION_FRAGMENTO)
    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    fragmentos = []
    for i in range(n_frags):
        offset     = i * DURACION_FRAGMENTO
        frag_path  = str(Path(carpeta_salida) / f"fragmento_{i:03d}.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(offset), "-t", str(DURACION_FRAGMENTO),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                frag_path,
            ],
            capture_output=True,
        )
        fragmentos.append((frag_path, offset))
    print(f"Audio dividido en {len(fragmentos)} fragmento(s).")
    return fragmentos


def transcribir(client: OpenAI, audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=MODELO_TRANSCRIPCION,
            file=f,
            language="es",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return resp.model_dump()


def transcribir_completo(client: OpenAI, audio_path: str, carpeta_temp: str) -> list:
    tamano = os.path.getsize(audio_path)
    segmentos = []
    if tamano <= LIMITE_API_BYTES:
        print("Transcribiendo audio (archivo único)...")
        resultado = transcribir(client, audio_path)
        segmentos.extend(resultado.get("segments") or [])
    else:
        print(f"Audio pesa {tamano / 1024 / 1024:.1f} MB — dividiendo en fragmentos...")
        fragmentos = dividir_audio(audio_path, carpeta_temp)
        for idx, (frag_path, offset) in enumerate(fragmentos, 1):
            print(f"  Transcribiendo fragmento {idx}/{len(fragmentos)} (offset {offset}s)...")
            resultado = transcribir(client, frag_path)
            for seg in (resultado.get("segments") or []):
                seg["start"] = (seg.get("start") or 0) + offset
                seg["end"]   = (seg.get("end")   or 0) + offset
                segmentos.append(seg)
    return segmentos


def formatear_tiempo(segundos: float) -> str:
    total  = int(round(segundos))
    horas  = total // 3600
    minutos = (total % 3600) // 60
    segs   = total % 60
    if horas > 0:
        return f"{horas:02d}:{minutos:02d}:{segs:02d}"
    return f"{minutos:02d}:{segs:02d}"


def detectar_palabras_clave(segmentos: list) -> list:
    detecciones = []
    for seg in segmentos:
        texto = (seg.get("text") or "").strip()
        if not texto:
            continue
        texto_lower = texto.lower()
        encontradas = []
        for patron, etiqueta in PATRONES_CLAVE.items():
            if re.search(patron, texto_lower, flags=re.IGNORECASE):
                if etiqueta not in encontradas:
                    encontradas.append(etiqueta)
        if encontradas:
            detecciones.append({
                "palabras":    encontradas,
                "oracion":     texto,
                "inicio":      seg["start"],
                "fin":         seg["end"],
                "inicio_fmt":  formatear_tiempo(seg["start"]),
                "fin_fmt":     formatear_tiempo(seg["end"]),
            })
    return detecciones


def main():
    # Une todos los argumentos para manejar rutas con espacios
    if len(sys.argv) < 2:
        print("Uso: python detectar_palabras.py <ruta_al_video>")
        sys.exit(1)

    video_path = " ".join(sys.argv[1:])
    if not Path(video_path).exists():
        print(f"Error: no se encontró el archivo '{video_path}'")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: define la variable de entorno OPENAI_API_KEY.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    carpeta_trabajo = Path("trabajo_temp")
    carpeta_trabajo.mkdir(exist_ok=True)
    audio_path = str(carpeta_trabajo / "audio_completo.wav")

    extraer_audio(video_path, audio_path)

    segmentos = transcribir_completo(client, audio_path, str(carpeta_trabajo / "fragmentos"))
    print(f"\nTotal de segmentos transcritos: {len(segmentos)}")

    with open("transcripcion_completa.json", "w", encoding="utf-8") as f:
        json.dump(segmentos, f, ensure_ascii=False, indent=2)
    print("Transcripción guardada en 'transcripcion_completa.json'")

    detecciones = detectar_palabras_clave(segmentos)

    print(f"\n{'='*60}")
    print(f"PALABRAS INTERROGATIVAS DETECTADAS: {len(detecciones)}")
    print(f"{'='*60}\n")
    for d in detecciones:
        print(f"[{d['inicio_fmt']} - {d['fin_fmt']}]  ({', '.join(d['palabras'])})")
        print(f"    \"{d['oracion']}\"\n")

    with open("detecciones.json", "w", encoding="utf-8") as f:
        json.dump(detecciones, f, ensure_ascii=False, indent=2)
    print("Resultados guardados en 'detecciones.json'")


if __name__ == "__main__":
    main()
