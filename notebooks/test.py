"""
test.py — Detección de señas WH en tiempo real con webcam.

Uso:
    python notebooks/test.py
    python notebooks/test.py --modelo notebooks/models/svm_lsp.pkl
    python notebooks/test.py --camara 1   # si tienes varias cámaras

Controles:
    Q / ESC → salir
    R       → resetear el buffer de frames
    ESPACIO → pausar/reanudar
"""

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── Configuración ──────────────────────────────────────────────────────────────
# Ruta relativa al propio script — funciona desde cualquier directorio
MODEL_PATH    = Path(__file__).parent / 'models' / 'svm_lsp.pkl'
N_FRAMES      = 60       # debe coincidir con el entrenamiento
VENTANA_SEG   = 2.0      # segundos de video a acumular antes de predecir
UMBRAL_CONF   = 0.60     # confianza mínima para mostrar predicción
PRED_CADA_N   = 15       # predecir cada N frames nuevos (no en cada frame)
HISTORIAL_N   = 5        # últimas N predicciones para suavizar

# Colores BGR
COLOR_DERECHA  = (183, 74, 83)   # púrpura
COLOR_IZQ      = (86, 110, 15)   # verde
COLOR_PRED     = (255, 255, 255)
COLOR_FONDO    = (40, 40, 40)
COLOR_CONF_OK  = (100, 220, 100)
COLOR_CONF_LOW = (100, 150, 255)


# ── Inicializar MediaPipe ──────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ── Funciones ──────────────────────────────────────────────────────────────────

def limpiar_etiqueta(etiqueta: str) -> str:
    """Convierte 'QUE1' → 'QUE', 'QUIEN1' → 'QUIEN', etc."""
    return etiqueta.rstrip('0123456789')


def cargar_modelo(model_path: Path) -> dict:
    """Carga el bundle del modelo guardado en el notebook de entrenamiento."""
    if not model_path.exists():
        raise FileNotFoundError(
            f'No se encontró el modelo en {model_path}\n'
            'Ejecuta primero el notebook 04_entrenamiento.ipynb'
        )
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    print(f'Modelo cargado: {model_path.name}')
    print(f'  Clases: {bundle["clases"]}')
    print(f'  CV accuracy: {bundle["cv_accuracy"]:.3f}')
    print(f'  N frames: {bundle["n_frames"]}')
    return bundle


def extraer_landmarks_frame(resultado) -> np.ndarray:
    """
    Extrae los landmarks de un resultado de MediaPipe.
    Devuelve array (2, 21, 3) — mano derecha [0], izquierda [1].
    Rellena con ceros si no detecta una mano.
    """
    frame_lm = np.zeros((2, 21, 3), dtype=np.float32)
    if resultado.multi_hand_landmarks:
        for hand_lm, handedness in zip(
            resultado.multi_hand_landmarks,
            resultado.multi_handedness
        ):
            idx = 0 if handedness.classification[0].label == 'Right' else 1
            for j, lm in enumerate(hand_lm.landmark):
                frame_lm[idx, j] = [lm.x, lm.y, lm.z]
    return frame_lm


def normalizar_longitud(secuencia: np.ndarray, n_frames: int) -> np.ndarray:
    """Interpola la secuencia a exactamente n_frames."""
    T = len(secuencia)
    if T == n_frames:
        return secuencia
    indices = np.linspace(0, T - 1, n_frames)
    resultado = np.zeros((n_frames, 2, 21, 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        i0 = int(idx)
        i1 = min(i0 + 1, T - 1)
        alpha = idx - i0
        resultado[i] = secuencia[i0] * (1 - alpha) + secuencia[i1] * alpha
    return resultado


def predecir(bundle: dict, buffer: deque) -> tuple[str, float] | None:
    """
    Corre el modelo sobre el buffer de frames acumulados.
    Devuelve (etiqueta_limpia, confianza) o None si el buffer está vacío.
    """
    if len(buffer) < 10:
        return None

    secuencia = np.array(list(buffer))              # (T, 2, 21, 3)
    secuencia = normalizar_longitud(secuencia, bundle['n_frames'])
    vector    = secuencia.flatten().reshape(1, -1)  # (1, N)

    probs     = bundle['pipeline'].predict_proba(vector)[0]
    idx_max   = probs.argmax()
    confianza = probs[idx_max]
    etiqueta  = limpiar_etiqueta(bundle['label_encoder'].classes_[idx_max])

    return etiqueta, confianza


def dibujar_landmarks(frame: np.ndarray, resultado) -> None:
    """Dibuja landmarks de manos encima del frame (in-place)."""
    if not resultado.multi_hand_landmarks:
        return
    for hand_lm, handedness in zip(
        resultado.multi_hand_landmarks,
        resultado.multi_handedness
    ):
        label = handedness.classification[0].label
        color = COLOR_DERECHA if label == 'Right' else COLOR_IZQ
        mp_draw.draw_landmarks(
            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=color, thickness=2),
        )


def predecir_suavizado(historial: deque) -> tuple[str, int] | None:
    """
    Vota sobre el historial de predicciones recientes.
    Devuelve (etiqueta_más_frecuente, votos) o None si el historial está vacío.
    Solo cuenta predicciones con confianza >= UMBRAL_CONF.
    """
    candidatos = [e for e, c in historial if c >= UMBRAL_CONF]
    if not candidatos:
        return None
    etiqueta = max(set(candidatos), key=candidatos.count)
    votos = candidatos.count(etiqueta)
    return etiqueta, votos


def dibujar_hud(frame: np.ndarray, estado: dict) -> None:
    """
    Dibuja el HUD (heads-up display) con la predicción y estado del buffer.
    Panel semi-transparente en la parte superior.
    """
    h, w = frame.shape[:2]
    # Patrón correcto: dibujar sobre frame, guardar copia limpia en overlay
    overlay = frame.copy()
    cv2.rectangle(frame, (0, 0), (w, 125), COLOR_FONDO, -1)
    cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)

    # Predicción principal (suavizada por historial)
    historial  = estado.get('historial', deque())
    suavizado  = predecir_suavizado(historial)
    pred       = estado.get('prediccion')
    confianza  = estado.get('confianza', 0.0)

    if suavizado:
        etiqueta_final, votos = suavizado
        color_conf  = COLOR_CONF_OK
        texto_pred  = f'{etiqueta_final}  ({confianza:.0%})'
    elif pred and confianza >= UMBRAL_CONF:
        color_conf = COLOR_CONF_OK
        texto_pred = f'{pred}  ({confianza:.0%})'
    elif pred:
        color_conf = COLOR_CONF_LOW
        texto_pred = f'{pred}?  ({confianza:.0%})'
    else:
        color_conf = (180, 180, 180)
        texto_pred = 'Acumulando frames...'

    cv2.putText(frame, texto_pred, (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color_conf, 2, cv2.LINE_AA)

    # Historial de predicciones recientes (pequeño, bajo la predicción)
    if historial:
        hist_texto = '  '.join(
            f'{e}({c:.0%})' for e, c in list(historial)[-4:]
        )
        cv2.putText(frame, hist_texto, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Buffer progress bar
    n_buf   = estado.get('n_buffer', 0)
    pct_buf = min(n_buf / N_FRAMES, 1.0)
    bar_w   = w - 40
    cv2.rectangle(frame, (20, 82), (20 + bar_w, 96), (80, 80, 80), -1)
    cv2.rectangle(frame, (20, 82),
                  (20 + int(bar_w * pct_buf), 96), (150, 120, 200), -1)
    cv2.putText(frame, f'Buffer: {n_buf}/{N_FRAMES} frames',
                (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    # FPS
    fps_texto = f'FPS: {estado.get("fps", 0):.0f}'
    cv2.putText(frame, fps_texto, (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    # Pausa
    if estado.get('pausado'):
        cv2.putText(frame, 'PAUSADO', (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3, cv2.LINE_AA)

    # Controles (esquina inferior)
    controles = 'Q/ESC: salir  |  R: resetear buffer  |  ESPACIO: pausar'
    cv2.putText(frame, controles, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


# ── Loop principal ─────────────────────────────────────────────────────────────

def main(model_path: Path, camara_idx: int) -> None:
    bundle = cargar_modelo(model_path)

    cap = cv2.VideoCapture(camara_idx)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la cámara {camara_idx}')

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30
    max_buf = int(fps_cam * VENTANA_SEG * 1.5)  # buffer ligeramente más grande
    buffer  = deque(maxlen=max_buf)

    estado = {
        'prediccion': None,
        'confianza':  0.0,
        'n_buffer':   0,
        'fps':        0.0,
        'pausado':    False,
        'historial':  deque(maxlen=HISTORIAL_N),
    }

    frames_desde_pred = 0
    t_prev = time.time()

    cv2.namedWindow('LSP — Detector de señas WH', cv2.WINDOW_NORMAL)

    print('\nCámara abierta. Mostrando ventana de detección...')
    print('Controles: Q/ESC = salir | R = resetear buffer | ESPACIO = pausar\n')

    while True:
        if not estado['pausado']:
            ok, frame = cap.read()
            if not ok:
                print('Error leyendo cámara')
                break

            frame = cv2.flip(frame, 1)  # espejo horizontal (más natural)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(frame_rgb)

            # Extraer y acumular landmarks
            lm_frame = extraer_landmarks_frame(resultado)
            buffer.append(lm_frame)
            frames_desde_pred += 1

            # Dibujar landmarks encima del frame
            dibujar_landmarks(frame, resultado)

            # Predecir cada PRED_CADA_N frames nuevos
            if frames_desde_pred >= PRED_CADA_N and len(buffer) >= 10:
                resultado_pred = predecir(bundle, buffer)
                if resultado_pred:
                    etiqueta, conf = resultado_pred
                    estado['prediccion'] = etiqueta
                    estado['confianza']  = conf
                    estado['historial'].append((etiqueta, conf))
                frames_desde_pred = 0

            # FPS
            t_now = time.time()
            estado['fps']      = 1.0 / max(t_now - t_prev, 1e-6)
            estado['n_buffer'] = len(buffer)
            t_prev = t_now

        # Dibujar HUD
        dibujar_hud(frame, estado)
        cv2.imshow('LSP — Detector de señas WH', frame)

        # Teclado
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q o ESC
            break
        elif key == ord('r'):       # R → resetear buffer
            buffer.clear()
            estado['prediccion'] = None
            estado['confianza']  = 0.0
            estado['historial'].clear()
            print('Buffer reseteado')
        elif key == ord(' '):       # ESPACIO → pausar
            estado['pausado'] = not estado['pausado']

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print('Demo cerrado.')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo cámara LSP')
    parser.add_argument('--modelo', type=Path, default=MODEL_PATH,
                        help='Ruta al modelo .pkl')
    parser.add_argument('--camara', type=int, default=0,
                        help='Índice de la cámara (default: 0)')
    args = parser.parse_args()

    main(args.modelo, args.camara)