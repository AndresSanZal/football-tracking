from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO

from team_classifier import TeamClassifier

VIDEO = "data/videos/match.mp4"
WEIGHTS = "models/yolo_4classes_best.pt"
TEAM_MODEL = "models/team_kmeans_siglip.joblib"  # o tu team_kmeans_siglip.joblib
PLAYER_ID = 2

def main():
    model = YOLO(WEIGHTS)
    team = TeamClassifier(model_path=TEAM_MODEL, device="cpu")

    frame_gen = sv.get_video_frames_generator(VIDEO)
    frame = next(frame_gen)

    res = model.predict(frame, imgsz=1280, conf=0.3, iou=0.7, verbose=False)[0]
    det = sv.Detections.from_ultralytics(res)
    players = det[det.class_id == PLAYER_ID].with_nms(0.5, class_agnostic=True)

    crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy][:60]  # hasta 60
    team_ids = team.predict(crops)

    print("Total crops:", len(crops))
    if len(team_ids) > 0:
        unique, counts = np.unique(team_ids, return_counts=True)
        print("Distribución equipos:", dict(zip(unique.tolist(), counts.tolist())))
        print("Primeros 20:", team_ids[:20])

if __name__ == "__main__":
    main()
