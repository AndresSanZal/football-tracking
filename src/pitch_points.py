from ultralytics import YOLO
import cv2
import numpy as np

field_model = YOLO("models/best2_pitch.pt")

def draw_pitch_keypoints(frame_to_draw, original_frame, field_model, conf_threshold=0.5):
    result = field_model(original_frame, verbose=False)[0]

    if result.keypoints is None:
        return frame_to_draw

    xy = result.keypoints.xy
    conf = result.keypoints.conf

    if xy is None or conf is None or len(xy) == 0:
        return frame_to_draw

    points = xy[0].cpu().numpy()
    scores = conf[0].cpu().numpy()

    for idx, ((x, y), score) in enumerate(zip(points, scores)):
        if score > conf_threshold:
            cx, cy = int(x), int(y)
            cv2.circle(frame_to_draw, (cx, cy), 6, (180, 20, 255), -1)
            cv2.putText(
                frame_to_draw,
                str(idx),
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    return frame_to_draw