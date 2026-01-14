import os
import yaml
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg("configs/config.yaml")

    video_path = cfg["video_path"]
    output_path = cfg["output_path"]
    weights = cfg["weights"]

    conf = float(cfg.get("conf", 0.3))
    ball_conf=float(cfg.get("ball_conf", 0.12))
    iou = float(cfg.get("iou", 0.5))
    max_frames = int(cfg.get("max_frames", 600))

    COCO_PERSON = int(cfg.get("coco_person_id", 0))
    COCO_BALL = int(cfg.get("coco_sports_ball_id", 32))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model
    model = YOLO(weights)

    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25,
        height=21,
        outline_thickness=1
    )

    # Tracker (solo personas)
    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_gen = sv.get_video_frames_generator(video_path)

    total = video_info.total_frames
    if max_frames > 0:
        total = min(total, max_frames)

    with sv.VideoSink(output_path, video_info) as sink:
        for i, frame in tqdm(enumerate(frame_gen), total=total):
            if max_frames > 0 and i >= max_frames:
                break

            # Inference
            result = model.predict(frame, conf=min(conf, ball_conf), iou=iou, imgsz=1280, verbose=False)[0]

            det = sv.Detections.from_ultralytics(result)

            ball = det[det.class_id == COCO_BALL]
            ball = ball[ball.confidence >= ball_conf]


            # Person detections (COCO person)
            persons = det[det.class_id == COCO_PERSON]

            # NMS extra (opcional)
            persons = persons.with_nms(threshold=iou, class_agnostic=True)

            # Track
            
            persons = tracker.update_with_detections(persons)

            labels = [f"#{tid}" for tid in persons.tracker_id]

            # Annotate
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(annotated, persons)
            annotated = label_annotator.annotate(annotated, persons, labels=labels)
            if len(ball) > 0:
                annotated = triangle_annotator.annotate(annotated, ball)

            sink.write_frame(annotated)

    print(f"\n✅ Exportado: {output_path}")


if __name__ == "__main__":
    main()
