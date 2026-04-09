


from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import supervision as sv
import torch
from transformers import AutoProcessor, SiglipVisionModel


class TeamClassifier:
    """
    Carga un artifact .joblib y predice team_id para crops.
    Soporta dos formatos de artifact:
      - con UMAP: {"siglip_name": ..., "reducer": UMAP, "kmeans": KMeans}
      - sin UMAP: {"siglip_name": ..., "kmeans": KMeans}  (KMeans entrenado en embeddings 768)
    """

    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 32):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.batch_size = batch_size

        artifact: Dict[str, Any] = joblib.load(self.model_path)

        self.siglip_name: str = artifact.get("siglip_name", "google/siglip-base-patch16-224")
        self.kmeans = artifact["kmeans"]
        self.reducer = artifact.get("reducer", None)  # puede no existir

        # Cargar SigLIP una sola vez (coste grande)
        self.processor = AutoProcessor.from_pretrained(self.siglip_name)
        self.model = SiglipVisionModel.from_pretrained(self.siglip_name).to(self.device)
        self.model.eval()

    def _extract_embeddings(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros((0, 768), dtype=np.float32)

        crops_pil = [sv.cv2_to_pillow(c) for c in crops_bgr]
        out = []

        with torch.no_grad():
            for i in range(0, len(crops_pil), self.batch_size):
                batch = crops_pil[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # (B, 768)
                out.append(emb)

        return np.concatenate(out, axis=0)

    def predict(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Devuelve team_id por crop.
        """
        if len(crops_bgr) == 0:
            return np.array([], dtype=int)

        embeddings = self._extract_embeddings(crops_bgr)  # (N, 768)

        # Si el artifact tiene reducer (UMAP), proyectamos a (N, 3)
        if self.reducer is not None:
            features = self.reducer.transform(embeddings)  # (N, 3)
        else:
            features = embeddings  # (N, 768)

        team_ids = self.kmeans.predict(features)
        return team_ids.astype(int)
