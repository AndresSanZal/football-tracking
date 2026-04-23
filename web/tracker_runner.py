"""tracker_runner.py
Lanza track_4classes.py como subproceso y actualiza la BD al terminar.
Se ejecuta desde BackgroundTasks de FastAPI (hilo separado).
"""

import json
import subprocess
import sys
from pathlib import Path

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
RUNS_DIR     = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Importar funciones de BD (mismo paquete)
import database as db


def run_tracking(match_id: int, video_path: str, team_model: str | None) -> None:
    """
    Lanza el pipeline de tracking y actualiza la BD con el resultado.
    Diseñado para ejecutarse en un hilo de BackgroundTasks.
    """
    db.set_match_processing(match_id)

    video_path  = Path(video_path)
    stem        = video_path.stem
    out_video_raw = RUNS_DIR / f"match_{match_id}_{stem}_raw.mp4"     # salida directa del tracker
    out_video     = RUNS_DIR / f"match_{match_id}_{stem}_annotated.mp4"  # remuxado para web
    out_json      = RUNS_DIR / f"match_{match_id}_{stem}_metrics.json"

    # Construir comando
    cmd = [
        sys.executable,
        str(SRC_DIR / "track_4classes.py"),
        "--source",  str(video_path),
        "--output",  str(out_video_raw),
        "--metrics_output", str(out_json),
    ]
    if team_model:
        cmd += ["--team_model", team_model]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            error = result.stderr[-2000:] if result.stderr else "Error desconocido"
            db.set_match_error(match_id, error)
            return

        # Remuxar con ffmpeg: mueve el moov atom al inicio para que el navegador
        # pueda reproducir el vídeo sin descargarlo completo (faststart).
        # También fuerza H.264 + AAC, los únicos codecs garantizados en todos los navegadores.
        ffmpeg_result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(out_video_raw),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(out_video),
            ],
            capture_output=True,
            text=True,
        )

        # Si ffmpeg falla (no instalado), usar el vídeo raw directamente
        if ffmpeg_result.returncode != 0 or not out_video.exists():
            out_video_raw.rename(out_video)
        else:
            out_video_raw.unlink(missing_ok=True)  # borrar el raw temporal

        # Leer FPS y total_frames del JSON generado
        fps, total_frames = 0.0, 0
        if out_json.exists():
            data = json.loads(out_json.read_text(encoding="utf-8"))
            fps          = data.get("fps", 0.0)
            total_frames = data.get("total_frames", 0)
            db.import_metrics(match_id, str(out_json))

        db.set_match_done(
            match_id,
            fps=fps,
            total_frames=total_frames,
            video_out=str(out_video),
        )

    except Exception as e:
        db.set_match_error(match_id, str(e))
