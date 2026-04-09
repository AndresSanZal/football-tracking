import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "sports"))
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
_FIELD_VERTICES = np.array(CONFIG.vertices, dtype=np.float32)

# If a new H moves test points more than this (cm) vs the previous H, reject it.
# 2000 cm = 20 m — generous enough for real camera movement, but catches garbage Hs.
MAX_DRIFT_CM = 2000


def _homography_drift(H_new: np.ndarray, H_prev: np.ndarray, frame_shape: tuple) -> float:
    """Max field-space drift (cm) when projecting mid-frame points through H_new vs H_prev."""
    h, w = frame_shape[:2]
    # Sample points spread across the frame where players typically appear
    pts = np.array([
        [w // 4, h // 2], [w // 2, h // 2], [3 * w // 4, h // 2],
        [w // 4, 2 * h // 3], [3 * w // 4, 2 * h // 3],
    ], dtype=np.float32).reshape(-1, 1, 2)
    p_new  = cv2.perspectiveTransform(pts, H_new).reshape(-1, 2)
    p_prev = cv2.perspectiveTransform(pts, H_prev).reshape(-1, 2)
    return float(np.linalg.norm(p_new - p_prev, axis=1).max())


class FieldCalibrator:
    def __init__(self, model_path: str, max_drift: float = MAX_DRIFT_CM):
        self.model = YOLO(model_path)
        self.max_drift = max_drift
        self.H_prev = None

    def get_homography(self, frame: np.ndarray):
        result = self.model(frame, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return self.H_prev

        kps  = result.keypoints.xy[0].cpu().numpy()    # (32, 2)
        conf = result.keypoints.conf[0].cpu().numpy()  # (32,)

        h, w = frame.shape[:2]
        visible = (
            (conf > 0.4) &
            (kps[:, 0] > 0) & (kps[:, 0] < w) &
            (kps[:, 1] > 0) & (kps[:, 1] < h)
        )

        if visible.sum() < 6:
            return self.H_prev

        field_pts = _FIELD_VERTICES[visible]
        frame_pts = kps[visible].astype(np.float32)

        H, mask = cv2.findHomography(
            frame_pts, field_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=15.0,
            confidence=0.995,
        )

        if H is None or (mask is not None and mask.ravel().sum() < 5):
            return self.H_prev

        # Validate: reject H if it differs too much from the previous one.
        # This prevents a single bad frame from corrupting H_prev and causing
        # the "goes crazy then recovers" pattern.
        if self.H_prev is not None:
            drift = _homography_drift(H, self.H_prev, frame.shape)
            if drift > self.max_drift:
                return self.H_prev  # silently discard garbage H

        self.H_prev = H
        return H

    def project(self, points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Map pixel coordinates (N, 2) → field coordinates in cm."""
        if H is None or len(points_xy) == 0:
            return np.empty((0, 2))
        pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
