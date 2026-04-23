import sys
import argparse
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent.parent / "sports"))
from team_classifier import TeamClassifier
from field_calibration import FieldCalibrator, CONFIG
from reid_reconnector import ReIDReconnector
from metrics_exporter import MetricsExporter
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

MINIMAP_SCALE = 0.05
# Tune FIELD_OFFSET (cm) to correct systematic projection bias.
# Positive x = shift right on minimap, positive y = shift down.
# Example: if dots appear ~150 cm too far right, set (-150, 0).
FIELD_OFFSET = (-500, -500)
POSITION_ALPHA = 0.07  # EMA weight for new position (lower = smoother dots)

BALL_KEYPOINTS_MAX = 60  # máximo de detecciones reales a recordar (no frames — el trail dura mucho más)
BALL_MAX_GAP_FRAMES = 15 # si entre dos detecciones consecutivas hay más de este nº de frames, no se conectan con línea
BALL_COLOR = (255, 255, 255)  # BGR white — dot at current position
BALL_MAX_JUMP_CM = 500   # positions farther than this from last valid are treated as no-detection
                         # at 30fps: 120km/h ball → ~111cm/frame, 200km/h → ~185cm/frame
BALL_MAX_JUMP_PX = 300   # pixel jump filter: detecciones >300px del último píxel válido = falsa detección
BALL_POSITION_ALPHA = 0.35  # EMA balón en campo (mayor que jugadores → más reactivo al movimiento real)
BALL_RESET_AFTER_FRAMES = 30  # si el balón lleva este nº de frames rechazados, se resetea el ancla
POSSESSION_DIST_CM = 200 #distancia máxima en cm para con siderar que un jugador tiene el balón
POSSESSION_HYSTERESIS = 10 #frames consecutivos necesarios para cambiar de equipo poseedor

calibrator = FieldCalibrator("models/best_pitch.pt")


# Dataset classes (YOLO):
# 0: ball
# 1: goalkeeper
# 2: player
# 3: referee
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

TRACK_IDS = {GOALKEEPER_ID, PLAYER_ID, REFEREE_ID}  # entidades a trackear


# Clases "de visualización" (cuando ya asignamos equipos)
TEAM0_CLASS = 0
TEAM1_CLASS = 1
REF_CLASS = 2


TEAM_COLORS = [
    sv.Color(0, 100, 255),    # Team 0 — blue
    sv.Color(255, 50, 0),     # Team 1 — red
    sv.Color(128, 128, 128),  # Referee — grey
]


def _overlay_minimap(frame: np.ndarray, minimap: np.ndarray, margin: int = 10) -> np.ndarray:
    """Paste minimap into the bottom-left corner of frame."""
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    # Clamp to frame bounds
    mh = min(mh, fh - margin)
    mw = min(mw, fw - margin)
    minimap = cv2.resize(minimap, (mw, mh))
    y1, x1 = fh - mh - margin, margin
    frame[y1:y1 + mh, x1:x1 + mw] = minimap
    return frame


def _smooth_field_positions(
    field_pos: np.ndarray,
    tracker_ids: np.ndarray,
    smooth_dict: dict,
    alpha: float,
) -> np.ndarray:
    """Per-tracker EMA on projected field positions to remove dot jitter."""
    result = field_pos.copy()
    for i, tid in enumerate(tracker_ids):
        if tid in smooth_dict:
            result[i] = alpha * field_pos[i] + (1 - alpha) * smooth_dict[tid]
        smooth_dict[tid] = result[i]
    return result


def _update_ball_trail(
    raw_pos: np.ndarray,          # (2,) field position, or None si no hay detección este frame
    last_valid_state: list,       # [last_valid_raw_pos or None] — para jump check en cm
    ema_state: list,              # [smoothed_pos or None] — EMA sobre posiciones aceptadas
    alpha: float,                 # EMA weight (0=sin actualizar, 1=sin suavizado)
    rejected_streak: list,        # [int] — frames consecutivos rechazados, para resetear el ancla
) -> tuple:
    """Valida la posición del balón y aplica EMA si es aceptada. NO gestiona ningún deque.
    Si lleva BALL_RESET_AFTER_FRAMES frames rechazados consecutivos, resetea el ancla para
    que el siguiente frame aceptado no quede bloqueado por una posición antigua.
    Returns (status_str, smoothed_pos_or_None).
    Status: 'accepted', 'gap_none', 'gap_bounds', 'gap_jump(Xcm)'."""
    if raw_pos is None:
        rejected_streak[0] += 1
        if rejected_streak[0] >= BALL_RESET_AFTER_FRAMES:
            last_valid_state[0] = None
        return "gap_none", None

    # Rechazar si la proyección cae fuera del campo (homografía corrupta en este frame)
    if not (0 <= raw_pos[0] <= CONFIG.length and 0 <= raw_pos[1] <= CONFIG.width):
        rejected_streak[0] += 1
        if rejected_streak[0] >= BALL_RESET_AFTER_FRAMES:
            last_valid_state[0] = None
        return "gap_bounds", None

    prev = last_valid_state[0]
    if prev is not None:
        jump = float(np.linalg.norm(raw_pos - prev))
        if jump > BALL_MAX_JUMP_CM:
            rejected_streak[0] += 1
            if rejected_streak[0] >= BALL_RESET_AFTER_FRAMES:
                last_valid_state[0] = None
            return f"gap_jump({jump:.0f}cm)", None

    # Posición aceptada: resetear contador y actualizar estados
    rejected_streak[0] = 0
    last_valid_state[0] = raw_pos
    if ema_state[0] is None:
        ema_state[0] = raw_pos.copy()
    else:
        ema_state[0] = alpha * raw_pos + (1 - alpha) * ema_state[0]
    return "accepted", ema_state[0]


def _draw_ball_keypoints(
    pitch: np.ndarray,
    keypoints: deque,   # deque de tuplas (frame_idx, np.ndarray(2,))
    scale: float,
    padding: int = 50,
) -> np.ndarray:
    """Dibuja la trayectoria del balón a partir de detecciones reales (sin gaps).
    - Conecta puntos consecutivos con línea solo si están a <= BALL_MAX_GAP_FRAMES frames de distancia.
    - El trazo se va desvaneciendo cuanto más antiguo es (fade por índice).
    - Dot blanco brillante en la posición más reciente."""
    pts = list(keypoints)  # lista de (frame_idx, pos)
    n = len(pts)
    if n < 1:
        return pitch

    for k in range(1, n):
        fidx_prev, pos_prev = pts[k - 1]
        fidx_curr, pos_curr = pts[k]
        # No conectar si el hueco temporal entre detecciones es demasiado grande
        if fidx_curr - fidx_prev > BALL_MAX_GAP_FRAMES:
            continue
        alpha = k / n  # 0 = más antiguo (tenue), 1 = más reciente (brillante)
        p1 = (int(pos_prev[0] * scale) + padding, int(pos_prev[1] * scale) + padding)
        p2 = (int(pos_curr[0] * scale) + padding, int(pos_curr[1] * scale) + padding)
        color = tuple(int(c * alpha) for c in BALL_COLOR)
        thickness = max(1, int(3 * alpha))
        cv2.line(pitch, p1, p2, color, thickness, cv2.LINE_AA)

    # Dot brillante en la posición más reciente
    _, last_pos = pts[-1]
    cx = int(last_pos[0] * scale) + padding
    cy = int(last_pos[1] * scale) + padding
    cv2.circle(pitch, (cx, cy), 5, BALL_COLOR, -1, cv2.LINE_AA)
    return pitch


def _debug_ball_overlay(
    frame: np.ndarray,
    ball_det: sv.Detections,
    ball_field_adj,     # np.ndarray (2,) posición ajustada en campo, o None
    trail_status: str,
    frame_idx: int,
) -> np.ndarray:
    """Dibuja info de debug del balón en el frame y lo imprime por consola.
    Verde = aceptado, rojo = rechazado."""
    color = (0, 255, 0) if trail_status == "accepted" else (0, 0, 255)
    if len(ball_det) > 0:
        conf = float(ball_det.confidence[0])
        cx, cy = ball_det.get_anchors_coordinates(sv.Position.CENTER)[0]
        x1, y1, x2, y2 = ball_det.xyxy[0].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        field_str = (
            f"({ball_field_adj[0]:.0f},{ball_field_adj[1]:.0f})"
            if ball_field_adj is not None else "no_proj"
        )
        print(f"[f{frame_idx:05d}] ball conf={conf:.2f} px=({cx:.0f},{cy:.0f})"
              f" field={field_str} → {trail_status}")
        cv2.putText(frame, f"ball:{trail_status} c={conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    else:
        print(f"[f{frame_idx:05d}] ball: no_detection → {trail_status}")
        cv2.putText(frame, "ball:no_det", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 2, cv2.LINE_AA)
    return frame


def _print_calibration_diag(raw_field_pos: np.ndarray) -> None:
    """Print projected field positions once so you can measure FIELD_OFFSET."""
    PAD = 50  # draw_pitch default padding
    print("\n=== CALIBRATION DIAGNOSTIC ===")
    print(f"Field: x=[0,12000] cm (length), y=[0,7000] cm (width)")
    print(f"Minimap scale={MINIMAP_SCALE} → 1 minimap px = {1/MINIMAP_SCALE:.0f} cm")
    print(f"  If dots are N px too far right:  FIELD_OFFSET = (-N/{MINIMAP_SCALE}, 0)")
    print(f"  If dots are N px too far down:   FIELD_OFFSET = (0, -N/{MINIMAP_SCALE})")
    print("Raw projected positions (before FIELD_OFFSET):")
    for pos in raw_field_pos[:6]:
        px = int(pos[0] * MINIMAP_SCALE) + PAD
        py = int(pos[1] * MINIMAP_SCALE) + PAD
        print(f"  field=({pos[0]:7.0f}, {pos[1]:7.0f}) cm  →  minimap px=({px:4d}, {py:4d})")
    print("==============================\n")


def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """
    players.class_id debe ser 0/1 (equipos) en este punto.
    Devuelve array de 0/1 para los porteros según cercanía a centroides.
    """
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    # Si no hay jugadores clasificados todavía, fallback: todo a 0
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    p_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_ids = players.class_id.astype(int)

    # Si solo hay un equipo presente en este frame, fallback
    if np.sum(team_ids == 0) == 0 or np.sum(team_ids == 1) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    c0 = p_xy[team_ids == 0].mean(axis=0)
    c1 = p_xy[team_ids == 1].mean(axis=0)

    out = []
    for xy in gk_xy:
        d0 = np.linalg.norm(xy - c0)
        d1 = np.linalg.norm(xy - c1)
        out.append(0 if d0 < d1 else 1)

    return np.array(out, dtype=int)

def _update_possession(
    ball_pos,           # np.ndarray (2,) posición del balón en campo, o None
    player_field_pos,   # np.ndarray (N, 2) posiciones de jugadores en campo
    player_teams,       # np.ndarray (N,) equipo de cada jugador (0 o 1)
    possession_current,     # [int or None] — se modifica en sitio
    possession_candidate,   # [int or None, int] — se modifica en sitio
    frame_idx: int = 0,
    debug: bool = False,
):
    """Actualiza SOLO quién tiene la posesión oficial (possession_current).
    La acumulación de possession_frames ocurre en main() cada frame,
    de forma que el contador del poseedor sigue corriendo durante pases
    (balón en el aire) hasta que el equipo contrario consigue su propio streak.

    Cuando ball_pos es None: congelar candidato, no modificar possession_current.
    Cuando closest_dist > POSSESSION_DIST_CM: resetear candidato (balón suelto),
      pero possession_current NO cambia — el equipo sigue "teniendo" el balón.
    """
    # Sin balón este frame: congelar estado, no acumular
    if ball_pos is None:
        if debug:
            print(f"[poss f{frame_idx:05d}] ball=None → estado congelado"
                  f" | candidato=T{possession_candidate[0]} streak={possession_candidate[1]}"
                  f" | oficial=T{possession_current[0]}")
        return

    # Sin jugadores este frame: igual, congelar
    if len(player_field_pos) == 0:
        if debug:
            print(f"[poss f{frame_idx:05d}] sin jugadores → estado congelado")
        return

    mask = (player_teams == 0) | (player_teams == 1)
    if not mask.any():
        if debug:
            print(f"[poss f{frame_idx:05d}] solo árbitros → estado congelado")
        return

    pos_filtered   = player_field_pos[mask]
    teams_filtered = player_teams[mask]

    # Jugador más cercano al balón
    dists        = np.linalg.norm(pos_filtered - ball_pos, axis=1)
    closest_idx  = int(np.argmin(dists))
    closest_dist = dists[closest_idx]
    closest_team = int(teams_filtered[closest_idx])

    # Si está demasiado lejos, el balón está suelto → resetear candidato
    if closest_dist > POSSESSION_DIST_CM:
        possession_candidate[0] = None
        possession_candidate[1] = 0
        if debug:
            print(f"[poss f{frame_idx:05d}] ball=detected  más cercano=T{closest_team}"
                  f" dist={closest_dist:.0f}cm > {POSSESSION_DIST_CM}cm → suelto"
                  f" | oficial=T{possession_current[0]}")
        return

    # Histéresis: acumular frames consecutivos del mismo candidato
    if possession_candidate[0] == closest_team:
        possession_candidate[1] += 1
    else:
        possession_candidate[0] = closest_team
        possession_candidate[1] = 1

    # Cambiar posesión oficial solo si llevamos suficientes frames consecutivos
    if possession_candidate[1] >= POSSESSION_HYSTERESIS:
        possession_current[0] = closest_team

    if debug:
        confirmed = " ✓ CONFIRMADO" if possession_candidate[1] >= POSSESSION_HYSTERESIS else ""
        print(f"[poss f{frame_idx:05d}] ball=detected  más cercano=T{closest_team}"
              f" dist={closest_dist:.0f}cm  streak={possession_candidate[1]}/{POSSESSION_HYSTERESIS}"
              f"{confirmed} | oficial=T{possession_current[0]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=str, default="models/yolo_4classes_best.pt")
    parser.add_argument("--output", type=str, default="runs/out_4classes_bytetrack.mp4")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--team_model", type=str, default=None)
    parser.add_argument("--debug_ball", action="store_true",
                        help="Imprime por consola el estado del balón cada frame y dibuja el bbox en el vídeo")
    parser.add_argument("--debug_possession", action="store_true",
                        help="Imprime por consola el estado de la posesión cada frame")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Ruta del JSON de métricas a exportar al finalizar (ej: runs/metrics.json)")
    args = parser.parse_args()

    source_path = Path(args.source)
    assert source_path.exists(), f"No existe el vídeo: {source_path}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    video_info = sv.VideoInfo.from_video_path(str(source_path))

    # Cargar clasificador de equipos (opcional)
    team_classifier = None
    if args.team_model:
        team_classifier = TeamClassifier(model_path=args.team_model, device="cpu")

    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)
    triangle_annotator = sv.TriangleAnnotator(base=25, height=21, outline_thickness=2)

    # Tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.30,
        lost_track_buffer=90,              # el video esta a 30 fps asi que le damos 3 segundos para encontrar id de nuevo
        minimum_matching_threshold=0.60,
        frame_rate=video_info.fps
    )
    tracker.reset()

    # ReID reconnector (persistencia de IDs)
    reconnector = ReIDReconnector()

    # Métricas (exportación JSON al final)
    metrics_out_path = args.metrics_output
    if metrics_out_path is None:
        stem = Path(args.source).stem
        metrics_out_path = str(Path(args.output).parent / f"metrics_{stem}.json")
    exporter = MetricsExporter(video_path=args.source, fps=video_info.fps)

    # Team smoothing (por track_id)
    TEAM_WINDOW = 15  # 15–30
    team_history = defaultdict(lambda: deque(maxlen=TEAM_WINDOW))
    field_pos_smooth: dict = {}  # tracker_id -> smoothed field position (np.ndarray)
    _diag_printed = False  # print field positions once to help calibrate FIELD_OFFSET
    ball_keypoints: deque = deque(maxlen=BALL_KEYPOINTS_MAX)  # tuplas (frame_idx, pos) — solo detecciones reales
    ball_smooth_state: list = [None]     # [last_valid_raw_pos or None] — para jump check en campo
    ball_ema_state: list = [None]        # [smoothed_field_pos or None] — EMA suavizado
    last_valid_ball_px: list = [None]    # [last_valid_pixel_center or None] — para filtro en píxeles
    ball_rejected_streak: list = [0]     # [int] — frames consecutivos rechazados; resetea el ancla si supera BALL_RESET_AFTER_FRAMES
    possession_frames = [0,0] #Lista -> frames acumulados de posesión por equipo
    possession_current = [None] #Lista -> equipo que tiene el balón, será 0, 1 o None
    possession_candidate = [None, 0] #Lista -> [equipo candidato a tener la posesión, frames consecutivos con ventaja]

    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        frame_generator = sv.get_video_frames_generator(str(source_path))

        for i, frame in enumerate(frame_generator):
            if args.max_frames > 0 and i >= args.max_frames:
                break

            # Inference
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # Conf por clase
            people_conf = 0.30
            ball_conf = 0.20

            is_ball = detections.class_id == BALL_ID
            keep_ball = is_ball & (detections.confidence >= ball_conf)
            keep_people = (~is_ball) & (detections.confidence >= people_conf)
            detections = detections[keep_ball | keep_people]

            # --- Ball (no tracking) ---
            ball_det = detections[detections.class_id == BALL_ID]
            if len(ball_det) > 0:
                best_idx = int(np.argmax(ball_det.confidence))
                ball_det = ball_det[[best_idx]]
                ball_det.xyxy = sv.pad_boxes(ball_det.xyxy, px=10)

            # --- People (tracking) ---
            people_det = detections[np.isin(detections.class_id, list(TRACK_IDS))]

            if len(people_det) > 0:
                people_det = people_det.with_nms(threshold=0.4, class_agnostic=True)
                people_det.xyxy = sv.pad_boxes(people_det.xyxy, px=8)

            tracked = tracker.update_with_detections(detections=people_det)

            # Si NO hay team_classifier, simplemente renderizamos lo trackeado como venía
            # (GK/P/R con ids) y seguimos.
            if team_classifier is None or len(tracked) == 0 or tracked.tracker_id is None:
                labels = []
                if len(tracked) > 0 and tracked.tracker_id is not None:
                    cls_map = {GOALKEEPER_ID: "GK", PLAYER_ID: "P", REFEREE_ID: "R"}
                    for cid, tid in zip(tracked.class_id, tracked.tracker_id):
                        labels.append(f"{cls_map.get(int(cid), 'X')} #{int(tid)}")

                annotated = frame.copy()
                H = calibrator.get_homography(frame)
                annotated = ellipse_annotator.annotate(scene=annotated, detections=tracked)
                if labels:
                    annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
                if len(ball_det) > 0:
                    annotated = triangle_annotator.annotate(scene=annotated, detections=ball_det)
                if H is not None:
                    if len(tracked) > 0:
                        feet = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        field_pos = calibrator.project(feet, H)
                        if not _diag_printed:
                            _print_calibration_diag(field_pos)
                            _diag_printed = True
                        field_pos += np.array(FIELD_OFFSET, dtype=np.float32)
                        field_pos = _smooth_field_positions(
                            field_pos, tracked.tracker_id, field_pos_smooth, POSITION_ALPHA
                        )
                    else:
                        field_pos = np.empty((0, 2))
                    # Sin team_classifier no sabemos a qué equipo pertenece cada jugador,
                    # así que no calculamos posesión en esta rama.
                    ball_field_adj = None
                    if len(ball_det) > 0:
                        ball_px_raw = ball_det.get_anchors_coordinates(sv.Position.CENTER)[0]
                        px_jump = (float(np.linalg.norm(ball_px_raw - last_valid_ball_px[0]))
                                   if last_valid_ball_px[0] is not None else 0.0)
                        if px_jump > BALL_MAX_JUMP_PX:
                            trail_status = f"px_jump({px_jump:.0f}px)"
                        else:
                            last_valid_ball_px[0] = ball_px_raw
                            ball_center = ball_det.get_anchors_coordinates(sv.Position.CENTER)
                            ball_field = calibrator.project(ball_center, H)
                            if len(ball_field) > 0:
                                ball_field += np.array(FIELD_OFFSET, dtype=np.float32)
                                ball_field_adj = ball_field[0]
                                trail_status, ball_pos_smooth = _update_ball_trail(ball_field_adj, ball_smooth_state, ball_ema_state, BALL_POSITION_ALPHA, ball_rejected_streak)
                                if trail_status == "accepted":
                                    ball_keypoints.append((i, ball_pos_smooth))
                            else:
                                trail_status, _ = _update_ball_trail(None, ball_smooth_state, ball_ema_state, BALL_POSITION_ALPHA, ball_rejected_streak)
                    else:
                        trail_status, _ = _update_ball_trail(None, ball_smooth_state, ball_ema_state, BALL_POSITION_ALPHA, ball_rejected_streak)
                    if args.debug_ball:
                        annotated = _debug_ball_overlay(annotated, ball_det, ball_field_adj, trail_status, i)

                    minimap = draw_pitch(CONFIG, scale=MINIMAP_SCALE)
                    if len(field_pos) > 0:
                        minimap = draw_points_on_pitch(
                            CONFIG, field_pos,
                            face_color=sv.Color.WHITE,
                            scale=MINIMAP_SCALE,
                            pitch=minimap,
                        )
                    if len(ball_keypoints) > 0:
                        minimap = _draw_ball_keypoints(minimap, ball_keypoints, MINIMAP_SCALE)
                    annotated = _overlay_minimap(annotated, minimap)

                sink.write_frame(annotated)
                continue


            # TEAM LOGIC (con team_classifier)

            # 1) Separar tracked por tipo ORIGINAL
            gk_det = tracked[tracked.class_id == GOALKEEPER_ID]
            pl_det = tracked[tracked.class_id == PLAYER_ID]
            ref_det = tracked[tracked.class_id == REFEREE_ID]

            # 2) Clasificar players a equipos (0/1) + smoothing por track_id
            #    Guardamos los embeddings SiGLIP para el ReID reconnector
            reid_embeddings: dict = {}   # {tracker_id: np.ndarray(768,)}
            if len(pl_det) > 0:
                pl_crops = [sv.crop_image(frame, xyxy) for xyxy in pl_det.xyxy]
                # Algunos crops pueden ser vacíos si el bbox queda fuera del frame
                valid_mask = np.array([c.size > 0 for c in pl_crops])
                if not valid_mask.any():
                    pl_det = pl_det[valid_mask]
                    pl_crops = []
                else:
                    pl_det = pl_det[valid_mask]
                    pl_crops = [c for c, v in zip(pl_crops, valid_mask) if v]

                if pl_crops:
                    pred_team, emb_matrix = team_classifier.predict_with_embeddings(pl_crops)
                    # Guardar embedding por tracker_id
                    for local_idx in range(len(pl_det)):
                        tid = int(pl_det.tracker_id[local_idx])
                        reid_embeddings[tid] = emb_matrix[local_idx]
                else:
                    pred_team = np.array([], dtype=int)

                # actualizar historial por tracker_id
                for local_idx, team_id in enumerate(pred_team):
                    tid = int(pl_det.tracker_id[local_idx])
                    team_history[tid].append(int(team_id))

                # aplicar suavizado al class_id de players
                smooth_ids = []
                for local_idx in range(len(pl_det)):
                    tid = int(pl_det.tracker_id[local_idx])
                    hist = team_history[tid]
                    # voto mayoritario (para 0/1 la media redondeada va bien)
                    smooth_team = int(round(float(np.mean(hist)))) if len(hist) else int(pred_team[local_idx])
                    smooth_ids.append(smooth_team)

                pl_det.class_id = np.array(smooth_ids, dtype=int)

            # 3) Asignar porteros al equipo más cercano (usa players ya 0/1)
            if len(gk_det) > 0:
                gk_det.class_id = resolve_goalkeepers_team_id(pl_det, gk_det)

            # 4) Árbitros -> clase fija REF_CLASS (=2)
            if len(ref_det) > 0:
                ref_det.class_id = np.full(len(ref_det), REF_CLASS, dtype=int)

            # 5) Unir de nuevo (ya en "clases de visualización")
            tracked_vis = sv.Detections.merge([pl_det, gk_det, ref_det])
            tracked_vis.class_id = tracked_vis.class_id.astype(int)

            # 6) ReID reconnector: reasignar IDs canónicos
            if tracked_vis.tracker_id is not None and len(tracked_vis.tracker_id) > 0:
                # Construir dicts para el reconnector
                reid_teams: dict = {
                    int(tid): int(cid)
                    for tid, cid in zip(tracked_vis.tracker_id, tracked_vis.class_id)
                }
                # field_pos_smooth aún no está actualizado para este frame,
                # pero usamos el valor del frame anterior (suficiente para gating)
                reid_fpos: dict = {
                    int(tid): field_pos_smooth[int(tid)]
                    for tid in tracked_vis.tracker_id
                    if int(tid) in field_pos_smooth
                }
                # Posición en píxeles (pies) para TTL diferenciado por borde
                reid_pxpos: dict = {
                    int(tid): coords
                    for tid, coords in zip(
                        tracked_vis.tracker_id,
                        tracked_vis.get_anchors_coordinates(sv.Position.BOTTOM_CENTER),
                    )
                }
                fh, fw = frame.shape[:2]
                id_map = reconnector.update(
                    frame_idx=i,
                    track_ids=tracked_vis.tracker_id,
                    embeddings=reid_embeddings,
                    teams=reid_teams,
                    field_pos=reid_fpos,
                    pixel_pos=reid_pxpos,
                    frame_wh=(fw, fh),
                )
                if id_map:
                    tracked_vis.tracker_id = np.array(
                        [id_map.get(int(t), int(t)) for t in tracked_vis.tracker_id],
                        dtype=tracked_vis.tracker_id.dtype,
                    )

                # Asignar IDs fijos 1-20 (T0: 1-10, T1: 11-20)
                tracked_vis.tracker_id = np.array(
                    [reconnector.get_fixed_id(int(t), int(c))
                     for t, c in zip(tracked_vis.tracker_id, tracked_vis.class_id)],
                    dtype=tracked_vis.tracker_id.dtype,
                )

            # Etiquetas
            cls_map = {TEAM0_CLASS: "T0", TEAM1_CLASS: "T1", REF_CLASS: "REF"}
            labels = [f"{cls_map.get(int(cid), 'X')} #{int(tid)}"
                      for cid, tid in zip(tracked_vis.class_id, tracked_vis.tracker_id)]

            annotated = frame.copy()
            H = calibrator.get_homography(frame)
            annotated = ellipse_annotator.annotate(scene=annotated, detections=tracked_vis)
            if labels:
                annotated = label_annotator.annotate(scene=annotated, detections=tracked_vis, labels=labels)
            if len(ball_det) > 0:
                annotated = triangle_annotator.annotate(scene=annotated, detections=ball_det)

            # Minimap overlay
            if H is not None and len(tracked_vis) > 0:
                feet = tracked_vis.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                field_pos = calibrator.project(feet, H)
                if not _diag_printed:
                    _print_calibration_diag(field_pos)
                    _diag_printed = True
                field_pos += np.array(FIELD_OFFSET, dtype=np.float32)
                field_pos = _smooth_field_positions(
                    field_pos, tracked_vis.tracker_id, field_pos_smooth, POSITION_ALPHA
                )
                ball_pos_now = None  # posición suavizada del balón (EMA) para posesión y minimap
                ball_field_adj = None
                if len(ball_det) > 0:
                    ball_px_raw = ball_det.get_anchors_coordinates(sv.Position.CENTER)[0]
                    px_jump = (float(np.linalg.norm(ball_px_raw - last_valid_ball_px[0]))
                               if last_valid_ball_px[0] is not None else 0.0)
                    if px_jump > BALL_MAX_JUMP_PX:
                        trail_status = f"px_jump({px_jump:.0f}px)"
                    else:
                        last_valid_ball_px[0] = ball_px_raw
                        ball_center = ball_det.get_anchors_coordinates(sv.Position.CENTER)
                        ball_field = calibrator.project(ball_center, H)
                        if len(ball_field) > 0:
                            ball_field += np.array(FIELD_OFFSET, dtype=np.float32)
                            ball_field_adj = ball_field[0]
                            trail_status, ball_pos_now = _update_ball_trail(ball_field_adj, ball_smooth_state, ball_ema_state, BALL_POSITION_ALPHA, ball_rejected_streak)
                            if trail_status == "accepted":
                                ball_keypoints.append((i, ball_pos_now))
                        else:
                            trail_status, _ = _update_ball_trail(None, ball_smooth_state, ball_ema_state, BALL_POSITION_ALPHA, ball_rejected_streak)
                else:
                    trail_status, _ = _update_ball_trail(None, ball_smooth_state, ball_ema_state, BALL_POSITION_ALPHA, ball_rejected_streak)
                if args.debug_ball:
                    annotated = _debug_ball_overlay(annotated, ball_det, ball_field_adj, trail_status, i)

                # Actualizar quién tiene la posesión oficial
                _update_possession(
                    ball_pos_now,
                    field_pos,
                    tracked_vis.class_id,
                    possession_current,
                    possession_candidate,
                    frame_idx=i,
                    debug=args.debug_possession,
                )
                # Acumular frames cada frame, independientemente de si el balón se detecta.
                # El poseedor sigue acumulando durante pases hasta que el equipo contrario
                # consiga su propio streak de POSSESSION_HYSTERESIS frames consecutivos.
                if possession_current[0] is not None:
                    possession_frames[possession_current[0]] += 1
                
                # Registrar métricas de este frame
                exporter.record_frame(
                    frame_idx=i,
                    track_ids=tracked_vis.tracker_id,
                    teams=tracked_vis.class_id,
                    field_pos=field_pos if len(field_pos) > 0 else None,
                    ball_pos=ball_pos_now,
                )

                pitch = draw_pitch(CONFIG, scale=MINIMAP_SCALE)
                if len(field_pos) > 0:
                    for cls_idx, color in enumerate(TEAM_COLORS):
                        mask = tracked_vis.class_id == cls_idx
                        if mask.any():
                            pitch = draw_points_on_pitch(
                                CONFIG, field_pos[mask],
                                face_color=color,
                                scale=MINIMAP_SCALE,
                                pitch=pitch,
                            )
                if len(ball_keypoints) > 0:
                    pitch = _draw_ball_keypoints(pitch, ball_keypoints, MINIMAP_SCALE)
                annotated = _overlay_minimap(annotated, pitch)

                # Barra de posesión encima del minimap
                total_frames = possession_frames[0] + possession_frames[1]
                if total_frames > 0:
                    pct0 = possession_frames[0] / total_frames * 100
                    pct1 = possession_frames[1] / total_frames * 100
                    fh, fw = annotated.shape[:2]
                    # Mismas dimensiones que usará _overlay_minimap al pegar el pitch
                    bar_w = min(pitch.shape[1], fw - 10)
                    bar_h = 28
                    bar_x = 10
                    mh_actual = min(pitch.shape[0], fh - 10)
                    bar_y = fh - mh_actual - 10 - bar_h - 4  # 4 px de separación con el minimap
                    # Fondo gris oscuro
                    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
                    # Relleno equipo 0 (azul) proporcional a su porcentaje
                    t0_w = int(bar_w * pct0 / 100)
                    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + t0_w, bar_y + bar_h), (255, 100, 0), -1)
                    # Relleno equipo 1 (rojo) ocupa el resto
                    cv2.rectangle(annotated, (bar_x + t0_w, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 50, 255), -1)
                    # Etiqueta T0 a la izquierda
                    cv2.putText(annotated, f"T0  {pct0:.0f}%", (bar_x + 6, bar_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    # Etiqueta T1 a la derecha (alineada al borde derecho de la barra)
                    label1 = f"{pct1:.0f}%  T1"
                    (lw1, _), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.putText(annotated, label1, (bar_x + bar_w - lw1 - 6, bar_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            sink.write_frame(annotated)


    print(f"Vídeo guardado en {out_path}")
    reconnector.print_stats()
    exporter.export(possession_frames=possession_frames, out_path=metrics_out_path)


if __name__ == "__main__":
    main()
