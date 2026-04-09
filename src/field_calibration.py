import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "sports"))
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
_FIELD_VERTICES = np.array(CONFIG.vertices, dtype=np.float32)


class FieldCalibrator:
    def __init__(self, model_path: str, alpha: float = 0.5):
        self.model = YOLO(model_path)
        self.alpha = alpha
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

        # Temporal smoothing — critical for dynamic/zooming camera
        if self.H_prev is not None:
            H = self.alpha * H + (1 - self.alpha) * self.H_prev

        self.H_prev = H
        return H

    def project(self, points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Map pixel coordinates (N, 2) → field coordinates in cm."""
        if H is None or len(points_xy) == 0:
            return np.empty((0, 2))
        pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
