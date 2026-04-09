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
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

MINIMAP_SCALE = 0.05
# Tune FIELD_OFFSET (cm) to correct systematic projection bias.
# Positive x = shift right on minimap, positive y = shift down.
# Example: if dots appear ~150 cm too far right, set (-150, 0).
FIELD_OFFSET = (-500, -500)
POSITION_ALPHA = 0.07  # EMA weight for new position (lower = smoother dots)

BALL_TRAIL_LEN = 60      # frames to keep in trail (~2 s at 30 fps)
BALL_COLOR = (255, 255, 255)  # BGR white — dot at current position
BALL_MAX_JUMP_CM = 500   # positions farther than this from last valid are treated as no-detection
                         # at 30fps: 120km/h ball → ~111cm/frame, 200km/h → ~185cm/frame

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
    raw_pos: np.ndarray,          # (2,) field position, or None if no detection this frame
    trail: deque,                 # deque of np.ndarray(2,) or np.ndarray(0,) for gaps
    last_valid_state: list,       # [last_valid_pos or None] — mutable single-element
) -> None:
    """Add ball position to trail, replacing outliers/missing frames with empty arrays (gaps).
    Gaps prevent _draw_ball_trail from connecting discontinuous detections with long lines."""
    if raw_pos is None:
        trail.append(np.empty(0))
        return

    # Reject if projected outside field bounds (bad homography on that frame)
    if not (0 <= raw_pos[0] <= CONFIG.length and 0 <= raw_pos[1] <= CONFIG.width):
        trail.append(np.empty(0))
        return

    prev = last_valid_state[0]
    if prev is not None:
        jump = float(np.linalg.norm(raw_pos - prev))
        if jump > BALL_MAX_JUMP_CM:
            trail.append(np.empty(0))
            return

    last_valid_state[0] = raw_pos
    trail.append(raw_pos.copy())


def _draw_ball_trail(pitch: np.ndarray, trail: deque, scale: float, padding: int = 50) -> np.ndarray:
    """Draw ball trail on pitch: fading line + bright dot at current position.
    Empty arrays in the trail act as gaps — no line is drawn across them."""
    pts = list(trail)
    n = len(pts)
    if n < 1:
        return pitch

    for k in range(1, n):
        # Skip segment if either endpoint is a gap (empty array)
        if pts[k - 1].shape[0] == 0 or pts[k].shape[0] == 0:
            continue
        alpha = k / n  # 0 = oldest (transparent), 1 = newest (bright)
        p1 = (int(pts[k-1][0] * scale) + padding, int(pts[k-1][1] * scale) + padding)
        p2 = (int(pts[k][0]   * scale) + padding, int(pts[k][1]   * scale) + padding)
        color = tuple(int(c * alpha) for c in BALL_COLOR)
        thickness = max(1, int(3 * alpha))
        cv2.line(pitch, p1, p2, color, thickness, cv2.LINE_AA)

    # Bright dot at the most recent valid position
    last_valid = next((p for p in reversed(pts) if p.shape[0] > 0), None)
    if last_valid is not None:
        cx = int(last_valid[0] * scale) + padding
        cy = int(last_valid[1] * scale) + padding
        cv2.circle(pitch, (cx, cy), 5, BALL_COLOR, -1, cv2.LINE_AA)
    return pitch


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
        minimum_matching_threshold=0.70,
        frame_rate=video_info.fps
    )
    tracker.reset()

    # Team smoothing (por track_id)
    TEAM_WINDOW = 15  # 15–30
    team_history = defaultdict(lambda: deque(maxlen=TEAM_WINDOW))
    field_pos_smooth: dict = {}  # tracker_id -> smoothed field position (np.ndarray)
    _diag_printed = False  # print field positions once to help calibrate FIELD_OFFSET
    ball_trail: deque = deque(maxlen=BALL_TRAIL_LEN)  # field positions of the ball
    ball_smooth_state: list = [None]  # [prev_smoothed_pos or None]

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
                    if len(ball_det) > 0:
                        ball_center = ball_det.get_anchors_coordinates(sv.Position.CENTER)
                        ball_field = calibrator.project(ball_center, H)
                        if len(ball_field) > 0:
                            ball_field += np.array(FIELD_OFFSET, dtype=np.float32)
                            _update_ball_trail(ball_field[0], ball_trail, ball_smooth_state)
                        else:
                            _update_ball_trail(None, ball_trail, ball_smooth_state)
                    else:
                        _update_ball_trail(None, ball_trail, ball_smooth_state)
                    minimap = draw_pitch(CONFIG, scale=MINIMAP_SCALE)
                    if len(field_pos) > 0:
                        minimap = draw_points_on_pitch(
                            CONFIG, field_pos,
                            face_color=sv.Color.WHITE,
                            scale=MINIMAP_SCALE,
                            pitch=minimap,
                        )
                    minimap = _draw_ball_trail(minimap, ball_trail, MINIMAP_SCALE)
                    annotated = _overlay_minimap(annotated, minimap)
                sink.write_frame(annotated)
                continue


            # TEAM LOGIC (con team_classifier)

            # 1) Separar tracked por tipo ORIGINAL
            gk_det = tracked[tracked.class_id == GOALKEEPER_ID]
            pl_det = tracked[tracked.class_id == PLAYER_ID]
            ref_det = tracked[tracked.class_id == REFEREE_ID]

            # 2) Clasificar players a equipos (0/1) + smoothing por track_id
            if len(pl_det) > 0:
                pl_crops = [sv.crop_image(frame, xyxy) for xyxy in pl_det.xyxy]
                pred_team = team_classifier.predict(pl_crops)  # 0/1

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
                if len(ball_det) > 0:
                    ball_center = ball_det.get_anchors_coordinates(sv.Position.CENTER)
                    ball_field = calibrator.project(ball_center, H)
                    if len(ball_field) > 0:
                        ball_field += np.array(FIELD_OFFSET, dtype=np.float32)
                        _update_ball_trail(ball_field[0], ball_trail, ball_smooth_state)
                    else:
                        _update_ball_trail(None, ball_trail, ball_smooth_state)
                else:
                    _update_ball_trail(None, ball_trail, ball_smooth_state)
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
                pitch = _draw_ball_trail(pitch, ball_trail, MINIMAP_SCALE)
                annotated = _overlay_minimap(annotated, pitch)

            sink.write_frame(annotated)


    print(f"Vídeo guardado en {out_path}")


if __name__ == "__main__":
    main()
