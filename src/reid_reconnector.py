"""reid_reconnector.py
Reconexión online de tracklets fragmentados usando embeddings SiGLIP + gating espacial.

Mejoras sobre la versión inicial:
  - Bug fix: los contadores elim/reid solo cuentan fusiones efectivas nuevas.
  - Predicción de posición: cuando un track muere, se estima su velocidad media
    y se usa para predecir dónde estaría al comparar con un nuevo track.
  - TTL diferenciado por zona: un track que muere cerca del borde del frame
    (probable pan de cámara) recibe el doble de TTL que uno que muere en el centro.

Uso en el bucle principal:
    reconnector = ReIDReconnector()

    id_map = reconnector.update(
        frame_idx  = i,
        track_ids  = tracked_vis.tracker_id,
        embeddings = emb_per_track,           # {tid: np.ndarray(768,)}
        teams      = team_per_track,          # {tid: int 0/1/2}
        field_pos  = field_pos_per_track,     # {tid: np.ndarray(2,)}
        pixel_pos  = pixel_pos_per_track,     # {tid: np.ndarray(2,)} — pies en px
        frame_wh   = (frame_width, frame_height),
    )
    if tracked_vis.tracker_id is not None:
        tracked_vis.tracker_id = np.array(
            [id_map.get(int(t), int(t)) for t in tracked_vis.tracker_id]
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Hiperparámetros ────────────────────────────────────────────────────────────
TEAM_SLOT_SIZE    = 10    # slots por equipo: T0 → 1..10, T1 → 11..20
DEAD_TTL_CENTER   = 150   # frames de TTL para tracks que mueren en el centro (~5 s)
DEAD_TTL_EDGE     = 300   # frames de TTL para tracks que mueren en el borde (~10 s)
EDGE_MARGIN_PX    = 80    # píxeles desde el borde del frame para considerar "edge"
COST_THRESH       = 0.45  # coste máximo para aceptar reconexión por Hungarian
W_APP             = 0.65  # peso apariencia (cosine distance)
W_SPAT            = 0.35  # peso espacial
MAX_DIST_CM       = 5000  # distancia máxima en campo para gating espacial (50 m)
EMB_ALPHA         = 0.20  # EMA embedding (0=sin actualizar, 1=sin memoria)
MIN_OBS           = 3     # mínimo de frames observados para entrar en dead buffer
VEL_HISTORY       = 10    # frames hacia atrás para calcular velocidad media


@dataclass
class _TrackState:
    """Estado de un track (vivo o muerto)."""
    canonical_id:   int
    mean_emb:       np.ndarray                      # (768,) float32
    team:           int                             # 0, 1 ó 2
    last_field_pos: Optional[np.ndarray]            # (2,) cm
    last_frame:     int
    obs_count:      int = 0
    ttl:            int = DEAD_TTL_CENTER           # asignado al morir
    mean_velocity:  Optional[np.ndarray] = None     # (2,) cm/frame — estimada al morir
    # Historial de posiciones para calcular velocidad (solo mientras vive)
    pos_history:    List[Tuple[int, np.ndarray]] = field(default_factory=list)


class ReIDReconnector:
    """Mantiene el buffer de tracks muertos y resuelve reconexiones frame a frame."""

    def __init__(self) -> None:
        self._live:  Dict[int, _TrackState] = {}
        self._dead:  List[_TrackState]      = []
        self._remap: Dict[int, int]         = {}   # id_bytetrack → id_canónico

        self._all_raw_ids:       set[int] = set()
        self._all_canonical_ids: set[int] = set()
        self._elim_matches:      int      = 0   # fusiones efectivas por eliminación
        self._reid_matches:      int      = 0   # fusiones efectivas por Hungarian

        # Mapeo canónico → ID fijo (1-10 T0, 11-20 T1)
        self._fixed_id:  Dict[int, int] = {}
        self._next_slot: Dict[int, int] = {0: 1, 1: TEAM_SLOT_SIZE + 1}

    # ──────────────────────────────────────────────────────────────────────────
    def update(
        self,
        frame_idx:  int,
        track_ids:  np.ndarray,
        embeddings: Dict[int, np.ndarray],
        teams:      Dict[int, int],
        field_pos:  Dict[int, np.ndarray],
        pixel_pos:  Optional[Dict[int, np.ndarray]] = None,
        frame_wh:   Optional[Tuple[int, int]]       = None,
    ) -> Dict[int, int]:
        """Actualiza el estado y devuelve {id_bytetrack → id_canónico}."""
        current_ids = set(int(t) for t in track_ids) if track_ids is not None else set()
        self._all_raw_ids.update(current_ids)

        # 1) Tracks que han muerto este frame
        dead_ids = set(self._live.keys()) - current_ids
        for tid in dead_ids:
            state = self._live.pop(tid)
            if state.obs_count >= MIN_OBS:
                state.mean_velocity = self._compute_velocity(state.pos_history)
                state.ttl = self._compute_ttl(tid, pixel_pos, frame_wh)
                state.pos_history = []   # liberar memoria
                self._dead.append(state)

        # 2) Limpiar dead buffer por TTL individual
        self._dead = [s for s in self._dead if frame_idx - s.last_frame <= s.ttl]

        # 3) Tracks nuevos (no en _live)
        new_ids = current_ids - set(self._live.keys())

        # 4) Intentar reconectar
        if new_ids and self._dead:
            self._reconnect(frame_idx, new_ids, embeddings, teams, field_pos)

        # 5) Actualizar o crear estado de tracks vivos
        for tid in current_ids:
            emb  = embeddings.get(tid)
            team = teams.get(tid, 2)
            fpos = field_pos.get(tid)

            if tid in self._live:
                state = self._live[tid]
                if emb is not None:
                    state.mean_emb = EMB_ALPHA * emb + (1 - EMB_ALPHA) * state.mean_emb
                state.team = team
                if fpos is not None:
                    state.last_field_pos = fpos
                    state.pos_history.append((frame_idx, fpos.copy()))
                    if len(state.pos_history) > VEL_HISTORY:
                        state.pos_history.pop(0)
                state.last_frame = frame_idx
                state.obs_count += 1
            else:
                if emb is None:
                    emb = np.zeros(768, dtype=np.float32)
                canonical = self._remap.get(tid, tid)
                self._live[tid] = _TrackState(
                    canonical_id=canonical,
                    mean_emb=emb.copy(),
                    team=team,
                    last_field_pos=fpos,
                    last_frame=frame_idx,
                    obs_count=1,
                    pos_history=[(frame_idx, fpos.copy())] if fpos is not None else [],
                )

        # 6) Construir dict de remapping y registrar IDs canónicos
        result: Dict[int, int] = {}
        for tid in current_ids:
            state = self._live.get(tid)
            if state is not None:
                self._all_canonical_ids.add(state.canonical_id)
                if state.canonical_id != tid:
                    result[tid] = state.canonical_id
            else:
                self._all_canonical_ids.add(tid)
        return result

    # ── Estadísticas ──────────────────────────────────────────────────────────
    def print_stats(self) -> None:
        raw   = len(self._all_raw_ids)
        canon = len(self._all_canonical_ids)
        fused = raw - canon
        pct   = fused / raw * 100 if raw else 0.0
        total_matches = self._elim_matches + self._reid_matches
        print("\n── ReID Reconnector ──────────────────────────────")
        print(f"  IDs emitidos por ByteTrack (sin ReID): {raw}")
        print(f"  IDs canónicos tras reconexión:          {canon}")
        print(f"  Fragmentos fusionados:                  {fused}  ({pct:.1f}% reducción)")
        print(f"  Fusiones efectivas totales:             {total_matches}")
        print(f"    · por eliminación (1 candidato):      {self._elim_matches}")
        print(f"    · por ReID/Hungarian:                 {self._reid_matches}")
        print("──────────────────────────────────────────────────\n")

    # ── IDs fijos 1-20 ────────────────────────────────────────────────────────
    def get_fixed_id(self, canonical_id: int, team: int) -> int:
        """Devuelve el ID fijo (1-20). Árbitros u overflow → canonical_id sin tocar."""
        if team not in (0, 1):
            return canonical_id

        if canonical_id in self._fixed_id:
            return self._fixed_id[canonical_id]

        max_slot = TEAM_SLOT_SIZE * (team + 1)
        if self._next_slot[team] > max_slot:
            return canonical_id  # overflow por sustituciones o errores

        fixed = self._next_slot[team]
        self._fixed_id[canonical_id] = fixed
        self._next_slot[team] += 1
        return fixed

    # ── Helpers privados ──────────────────────────────────────────────────────
    def _apply_match(self, tid_new: int, dead_state: _TrackState) -> bool:
        """Registra la reconexión. Devuelve True si es una fusión efectiva nueva."""
        canonical = dead_state.canonical_id
        is_effective = (canonical != tid_new) and (tid_new not in self._remap)
        self._remap[tid_new] = canonical
        self._live[tid_new] = _TrackState(
            canonical_id=canonical,
            mean_emb=dead_state.mean_emb.copy(),
            team=dead_state.team,
            last_field_pos=dead_state.last_field_pos,
            last_frame=dead_state.last_frame,
            obs_count=dead_state.obs_count,
            mean_velocity=dead_state.mean_velocity,
        )
        return is_effective

    @staticmethod
    def _compute_velocity(
        pos_history: List[Tuple[int, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Velocidad media en cm/frame a partir del historial de posiciones."""
        if len(pos_history) < 2:
            return None
        f0, p0 = pos_history[0]
        f1, p1 = pos_history[-1]
        gap = f1 - f0
        if gap <= 0:
            return None
        return (p1 - p0) / gap  # (2,) cm/frame

    @staticmethod
    def _compute_ttl(
        tid: int,
        pixel_pos: Optional[Dict[int, np.ndarray]],
        frame_wh:  Optional[Tuple[int, int]],
    ) -> int:
        """TTL largo si el track murió cerca del borde del frame, normal si no."""
        if pixel_pos is None or frame_wh is None:
            return DEAD_TTL_CENTER
        px = pixel_pos.get(tid)
        if px is None:
            return DEAD_TTL_CENTER
        w, h = frame_wh
        x, y = float(px[0]), float(px[1])
        near_edge = (
            x < EDGE_MARGIN_PX or x > w - EDGE_MARGIN_PX or
            y < EDGE_MARGIN_PX or y > h - EDGE_MARGIN_PX
        )
        return DEAD_TTL_EDGE if near_edge else DEAD_TTL_CENTER

    def _predict_pos(self, dead: _TrackState, frame_idx: int) -> Optional[np.ndarray]:
        """Posición predicha del track muerto en frame_idx usando su velocidad."""
        if dead.last_field_pos is None:
            return None
        if dead.mean_velocity is None:
            return dead.last_field_pos
        gap = frame_idx - dead.last_frame
        return dead.last_field_pos + dead.mean_velocity * gap

    def _reconnect(
        self,
        frame_idx:  int,
        new_ids:    set[int],
        embeddings: Dict[int, np.ndarray],
        teams:      Dict[int, int],
        field_pos:  Dict[int, np.ndarray],
    ) -> None:
        """Reconexión en dos fases:
        1. Eliminación: 1 nuevo + 1 muerto del mismo equipo → match directo.
        2. Hungarian: casos ambiguos con coste apariencia + posición predicha.
        """
        # ── Fase 1: eliminación ───────────────────────────────────────────────
        new_by_team:  Dict[int, List[int]] = {}
        dead_by_team: Dict[int, List[int]] = {}  # índice en self._dead

        for tid in new_ids:
            new_by_team.setdefault(teams.get(tid, 2), []).append(tid)
        for j, dead in enumerate(self._dead):
            dead_by_team.setdefault(dead.team, []).append(j)

        matched_new:  set[int] = set()
        matched_dead: set[int] = set()

        for team, new_tids in new_by_team.items():
            dead_idxs = dead_by_team.get(team, [])
            if len(new_tids) == 1 and len(dead_idxs) == 1:
                effective = self._apply_match(new_tids[0], self._dead[dead_idxs[0]])
                if effective:
                    self._elim_matches += 1
                matched_new.add(new_tids[0])
                matched_dead.add(dead_idxs[0])

        self._dead = [s for k, s in enumerate(self._dead) if k not in matched_dead]

        # ── Fase 2: Hungarian ─────────────────────────────────────────────────
        new_list  = [t for t in new_ids if t not in matched_new]
        dead_list = self._dead

        if not new_list or not dead_list:
            return

        n_new  = len(new_list)
        n_dead = len(dead_list)
        cost   = np.full((n_new, n_dead), fill_value=1.0)

        for i, tid in enumerate(new_list):
            emb_new  = embeddings.get(tid)
            team_new = teams.get(tid, 2)
            fpos_new = field_pos.get(tid)

            if emb_new is None:
                continue

            for j, dead in enumerate(dead_list):
                if team_new != dead.team:
                    continue

                # Posición predicha con velocidad
                pred_pos = self._predict_pos(dead, frame_idx)
                if fpos_new is not None and pred_pos is not None:
                    dist = float(np.linalg.norm(fpos_new - pred_pos))
                    if dist > MAX_DIST_CM:
                        continue
                    spat_cost = dist / MAX_DIST_CM
                else:
                    spat_cost = 0.5

                # Distancia coseno
                norm_new  = np.linalg.norm(emb_new)
                norm_dead = np.linalg.norm(dead.mean_emb)
                if norm_new < 1e-6 or norm_dead < 1e-6:
                    app_cost = 1.0
                else:
                    app_cost = 1.0 - float(
                        np.dot(emb_new, dead.mean_emb) / (norm_new * norm_dead)
                    )

                cost[i, j] = W_APP * app_cost + W_SPAT * spat_cost

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_dead_indices: set[int] = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= COST_THRESH:
                continue
            effective = self._apply_match(new_list[i], dead_list[j])
            if effective:
                self._reid_matches += 1
            matched_dead_indices.add(j)

        self._dead = [s for k, s in enumerate(self._dead) if k not in matched_dead_indices]
