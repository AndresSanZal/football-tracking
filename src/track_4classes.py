import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# Dataset classes:
# 0: ball
# 1: goalkeeper
# 2: player
# 3: referee
BALL_ID = 0
TRACK_IDS = {1, 2, 3}  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=str, default="models/yolo_4classes_best.pt")
    parser.add_argument("--output", type=str, default="runs/out_4classes_bytetrack.mp4")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max_frames", type=int, default=-1)
    args = parser.parse_args()

    

    source_path = Path(args.source)
    assert source_path.exists(), f"No existe el vídeo: {source_path}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    print(f"FPS del vídeo: {video_info.fps}")


    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)
    triangle_annotator = sv.TriangleAnnotator(base=25, height=21, outline_thickness=2)

    # Tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.30,  
        lost_track_buffer=90,             
        minimum_matching_threshold=0.70,   
        frame_rate=video_info.fps 
        )
    tracker.reset()
    

    # Video IO
    
    
    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        frame_generator = sv.get_video_frames_generator(str(source_path))

        for i, frame in enumerate(frame_generator):
            if args.max_frames > 0 and i >= args.max_frames:
                break

            
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            # conf por clase (ajustable)
            people_conf = 0.30
            ball_conf = 0.20

            is_ball = detections.class_id == BALL_ID
            keep_ball = is_ball & (detections.confidence >= ball_conf)
            keep_people = (~is_ball) & (detections.confidence >= people_conf)

            detections = detections[keep_ball | keep_people]


            
            ball_det = detections[detections.class_id == BALL_ID]
            if len(ball_det) > 0:
               
                best_idx = int(np.argmax(ball_det.confidence))
                ball_det = ball_det[[best_idx]]

                
                ball_det.xyxy = sv.pad_boxes(ball_det.xyxy, px=10)

           
            people_det = detections[np.isin(detections.class_id, list(TRACK_IDS))]

            if len(people_det) > 0:
                people_det = people_det.with_nms(threshold=0.4, class_agnostic=True)

           
            if len(people_det) > 0:
                people_det.xyxy = sv.pad_boxes(people_det.xyxy, px=8)  # prueba 6–12

            tracked = tracker.update_with_detections(detections=people_det)
            print(f"frame {i}: people_det={len(people_det)} tracked={len(tracked)}")
            # Labels de IDs
            labels = []
            if len(tracked) > 0 and tracked.tracker_id is not None:
                # Si quieres incluir clase + id:
                # cls_name = {1:"GK",2:"P",3:"R"}
                cls_map = {1: "GK", 2: "P", 3: "R"}
                for cid, tid in zip(tracked.class_id, tracked.tracker_id):
                    labels.append(f"{cls_map.get(int(cid), 'X')} #{int(tid)}")

            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(scene=annotated, detections=tracked)
            if len(labels) > 0:
                annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            if len(ball_det) > 0:
                annotated = triangle_annotator.annotate(scene=annotated, detections=ball_det)

            sink.write_frame(annotated)
            
    


    print(f"Vídeo guardado en {out_path}")


if __name__ == "__main__":
    main()
