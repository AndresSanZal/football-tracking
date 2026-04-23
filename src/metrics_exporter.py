"""metrics_exporter.py
Acumula métricas por jugador y balón durante el procesado del vídeo
y las serializa a JSON al finalizar.

El JSON resultante es el contrato de datos hacia la web:
  - Backend puede leerlo e insertar en BD.
  - Contiene trayectorias, distancias, velocidades y posesión.

Estructura del JSON:
{
  "video":        "match.mp4",
  "fps":          30.0,
  "total_frames": 2700,
  "players": {
    "3": {
      "team":          0,
      "trajectory":    [[frame, x_cm, y_cm], ...],
      "distance_m":    1250.4,
      "avg_speed_kmh": 12.3,
      "max_speed_kmh": 28.7
    },
    ...
  },
  "ball": {
    "trajectory": [[frame, x_cm, y_cm], ...]
  },
  "possession": {
    "team0_frames": 1450,
    "team1_frames":  980,
    "team0_pct":    59.7,
    "team1_pct":    40.3
  }
}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class MetricsExporter:
    def __init__(self, video_path: str, fps: float) -> None:
        self.video_name = Path(video_path).name
        self.fps = fps

        # {canonical_id: {"team": int, "traj": [(frame, x, y), ...]}}
        self._players: Dict[int, dict] = defaultdict(lambda: {"team": -1, "traj": []})
        self._ball_traj: List[Tuple[int, float, float]] = []
        self._total_frames = 0

    # ── Llamar cada frame ──────────────────────────────────────────────────────

    def record_frame(
        self,
        frame_idx: int,
        track_ids: Optional[np.ndarray],       # canonical IDs (ya remapeados)
        teams:     Optional[np.ndarray],       # class_id por track (0/1/2)
        field_pos: Optional[np.ndarray],       # (N, 2) cm
        ball_pos:  Optional[np.ndarray],       # (2,) cm ó None
    ) -> None:
        self._total_frames = max(self._total_frames, frame_idx + 1)

        if track_ids is not None and len(track_ids) > 0:
            for idx, tid in enumerate(track_ids):
                tid = int(tid)
                team = int(teams[idx]) if teams is not None else -1
                if field_pos is not None and idx < len(field_pos):
                    x, y = float(field_pos[idx][0]), float(field_pos[idx][1])
                    self._players[tid]["traj"].append((frame_idx, x, y))
                    # El equipo puede cambiar los primeros frames; guardamos el más reciente
                    self._players[tid]["team"] = team

        if ball_pos is not None:
            self._ball_traj.append((frame_idx, float(ball_pos[0]), float(ball_pos[1])))

    # ── Llamar al final del vídeo ──────────────────────────────────────────────

    def export(self, possession_frames: List[int], out_path: str) -> None:
        """Calcula métricas derivadas y escribe el JSON."""
        players_out = {}
        for pid, data in self._players.items():
            traj = data["traj"]
            dist_m, avg_spd, max_spd = self._compute_kinematics(traj)
            players_out[str(pid)] = {
                "team":          data["team"],
                "trajectory":    traj,
                "distance_m":    round(dist_m, 2),
                "avg_speed_kmh": round(avg_spd, 2),
                "max_speed_kmh": round(max_spd, 2),
            }

        total_poss = possession_frames[0] + possession_frames[1]
        possession_out = {
            "team0_frames": possession_frames[0],
            "team1_frames": possession_frames[1],
            "team0_pct": round(possession_frames[0] / total_poss * 100, 1) if total_poss else 0.0,
            "team1_pct": round(possession_frames[1] / total_poss * 100, 1) if total_poss else 0.0,
        }

        payload = {
            "video":        self.video_name,
            "fps":          self.fps,
            "total_frames": self._total_frames,
            "players":      players_out,
            "ball":         {"trajectory": self._ball_traj},
            "possession":   possession_out,
        }

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Métricas exportadas → {out}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compute_kinematics(
        self, traj: List[Tuple[int, float, float]]
    ) -> Tuple[float, float, float]:
        """Devuelve (distancia_m, velocidad_media_kmh, velocidad_max_kmh)."""
        if len(traj) < 2:
            return 0.0, 0.0, 0.0

        total_cm = 0.0
        speeds_kmh: List[float] = []
        dt = 1.0 / self.fps  # segundos entre frames consecutivos del track

        for k in range(1, len(traj)):
            f0, x0, y0 = traj[k - 1]
            f1, x1, y1 = traj[k]
            gap_frames = f1 - f0
            if gap_frames <= 0:
                continue
            dist_cm = float(np.hypot(x1 - x0, y1 - y0))
            elapsed_s = gap_frames * dt
            total_cm += dist_cm
            speed_ms  = (dist_cm / 100.0) / elapsed_s   # m/s
            speed_kmh = speed_ms * 3.6
            # Filtrar velocidades imposibles (>45 km/h) para no contaminar la media
            if speed_kmh <= 45.0:
                speeds_kmh.append(speed_kmh)

        dist_m   = total_cm / 100.0
        avg_spd  = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0
        max_spd  = float(np.max(speeds_kmh))  if speeds_kmh else 0.0
        return dist_m, avg_spd, max_spd
