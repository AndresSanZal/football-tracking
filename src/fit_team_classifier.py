# scripts/fit_team_classifier.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

import torch
from transformers import AutoProcessor, SiglipVisionModel

import umap
from sklearn.cluster import KMeans
import joblib


# Dataset classes (tu YOLO 4 clases):
# 0: ball
# 1: goalkeeper
# 2: player
# 3: referee
PLAYER_CLASS_ID = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True, help="Video de entrada")
    p.add_argument("--weights", type=str, default="models/yolo_4classes_best.pt", help="Pesos YOLO")
    p.add_argument("--stride", type=int, default=30, help="Saltos de frames (30=1fps si el vídeo es 30fps)")
    p.add_argument("--conf", type=float, default=0.30, help="Conf para detección")
    p.add_argument("--iou", type=float, default=0.7, help="IoU para predicción YOLO")
    p.add_argument("--nms", type=float, default=0.5, help="NMS threshold")
    p.add_argument("--max_crops", type=int, default=1500, help="Límite de crops para acelerar")
    p.add_argument("--siglip", type=str, default="google/siglip-base-patch16-224", help="Modelo SigLIP")
    p.add_argument("--batch", type=int, default=32, help="Batch embeddings")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo")
    p.add_argument("--out", type=str, default="models/team_classifier_2teams.joblib", help="Ruta de guardado")
    return p.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def crop_players_from_video(
    video_path: str,
    model: YOLO,
    stride: int,
    conf: float,
    iou: float,
    nms_th: float,
    max_crops: int,
) -> list[np.ndarray]:
    frame_gen = sv.get_video_frames_generator(video_path, stride=stride)

    crops: list[np.ndarray] = []
    for frame in tqdm(frame_gen, desc="Collecting player crops"):
        # YOLO inference sobre el frame
        res = model.predict(source=frame, conf=conf, iou=iou, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)

        # Nos quedamos con players
        det = det[det.class_id == PLAYER_CLASS_ID]

        # NMS para quitar duplicados
        if len(det) > 0:
            det = det.with_nms(threshold=nms_th, class_agnostic=True)

        # Crops
        for xyxy in det.xyxy:
            crop = sv.crop_image(frame, xyxy)
            # Filtrado básico por seguridad (evita crops microscópicos)
            h, w = crop.shape[:2]
            if h < 20 or w < 20:
                continue
            crops.append(crop)
            if 0 < max_crops <= len(crops):
                return crops

    return crops


def extract_siglip_embeddings(
    crops_bgr: list[np.ndarray],
    siglip_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    processor = AutoProcessor.from_pretrained(siglip_name)
    model = SiglipVisionModel.from_pretrained(siglip_name).to(device)
    model.eval()

    # BGR (cv2) -> RGB PIL-like usando supervision helper
    crops_pil = [sv.cv2_to_pillow(crop) for crop in crops_bgr]

    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(crops_pil), batch_size), desc="Extracting embeddings"):
            batch = crops_pil[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)

            outputs = model(**inputs)
            # outputs.last_hidden_state: (B, tokens, 768)
            # Promediamos tokens => (B, 768)
            embeds = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            all_embeds.append(embeds)

    return np.concatenate(all_embeds, axis=0)


def main():
    args = parse_args()
    video_path = str(Path(args.source))
    weights_path = str(Path(args.weights))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert Path(video_path).exists(), f"No existe el vídeo: {video_path}"
    assert Path(weights_path).exists(), f"No existen los pesos: {weights_path}"

    device = choose_device(args.device)
    print(f"✅ Device: {device}")

    # 1) Detectar y recortar jugadores (1 fps aprox)
    yolo = YOLO(weights_path)
    crops = crop_players_from_video(
        video_path=video_path,
        model=yolo,
        stride=args.stride,
        conf=args.conf,
        iou=args.iou,
        nms_th=args.nms,
        max_crops=args.max_crops,
    )

    if len(crops) < 50:
        raise RuntimeError(
            f"Demasiados pocos crops ({len(crops)}). "
            f"Prueba: bajar stride, bajar conf, o subir max_crops."
        )

    print(f"✅ Crops recogidos: {len(crops)}")

    # 2) SigLIP embeddings (N, 768)
    embeds = extract_siglip_embeddings(
        crops_bgr=crops,
        siglip_name=args.siglip,
        device=device,
        batch_size=args.batch,
    )
    print(f"✅ Embeddings shape: {embeds.shape}")

    # 3) UMAP (N, 3)
    reducer = umap.UMAP(n_components=3, random_state=42)
    proj = reducer.fit_transform(embeds)
    print(f"✅ Projections shape: {proj.shape}")

    # 4) KMeans (2 clusters = 2 equipos)
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
    clusters = kmeans.fit_predict(proj)

    # Info rápida
    unique, counts = np.unique(clusters, return_counts=True)
    print("✅ Cluster counts:", dict(zip(unique.tolist(), counts.tolist())))

    # 5) Guardar todo lo necesario para predecir luego
    artifact = {
        "siglip_name": args.siglip,
        "reducer": reducer,
        "kmeans": kmeans,
        "note": "Team classifier: SigLIP(768) -> UMAP(3) -> KMeans(2)",
    }
    joblib.dump(artifact, out_path)
    print(f"💾 Guardado en: {out_path}")

    print("\nSiguiente paso (fase online): cargar este joblib y hacer predict(crops) por frame.")


if __name__ == "__main__":
    main()
