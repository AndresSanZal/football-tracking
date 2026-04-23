"""main.py
Aplicación FastAPI — punto de entrada del servidor web.

Arrancar con:
    cd web
    uvicorn main:app --reload
"""

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import database as db
import tracker_runner as runner

# ── Configuración de rutas ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

PROJECT_ROOT = BASE_DIR.parent
TEAM_MODEL   = PROJECT_ROOT / "models" / "team_classifier_2teams.joblib"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Football Tracker")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/runs",   StaticFiles(directory=str(BASE_DIR / "runs")),   name="runs")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Inicializar BD al arrancar
db.init_db()


# ── Rutas ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    matches = db.list_matches()
    return templates.TemplateResponse(request, "index.html", {"matches": matches})


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    # Guardar el vídeo subido
    dest = UPLOADS_DIR / video.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    # Crear registro en BD
    match_id = db.create_match(video.filename)

    # Lanzar tracking en background
    team_model = str(TEAM_MODEL) if TEAM_MODEL.exists() else None
    background_tasks.add_task(
        runner.run_tracking, match_id, str(dest), team_model
    )

    return RedirectResponse(url=f"/match/{match_id}/processing", status_code=303)


@app.get("/match/{match_id}/processing")
async def processing(request: Request, match_id: int):
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    if match["status"] == "done":
        return RedirectResponse(url=f"/match/{match_id}")
    return templates.TemplateResponse(request, "processing.html", {"match": dict(match)})


@app.get("/match/{match_id}/status")
async def match_status(match_id: int):
    """Endpoint de polling — devuelve el estado actual del partido."""
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404)
    return JSONResponse({
        "status":    match["status"],
        "error_msg": match["error_msg"],
    })


@app.get("/match/{match_id}")
async def match_dashboard(request: Request, match_id: int):
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    if match["status"] != "done":
        return RedirectResponse(url=f"/match/{match_id}/processing")

    players   = db.get_players(match_id)
    possession = db.get_possession(match_id)

    return templates.TemplateResponse(request, "match.html", {
        "match":      dict(match),
        "players":    [dict(p) for p in players],
        "possession": dict(possession) if possession else None,
    })


@app.get("/match/{match_id}/video")
async def match_video(match_id: int):
    """Sirve el vídeo anotado directamente."""
    match = db.get_match(match_id)
    if not match or not match["video_out"]:
        raise HTTPException(status_code=404)
    video_path = Path(match["video_out"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Vídeo no encontrado en disco")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/api/match/{match_id}/trajectories")
async def api_trajectories(match_id: int):
    """Devuelve trayectorias de jugadores y balón como JSON para Chart.js."""
    players = db.get_players(match_id)
    result  = {}
    for p in players:
        traj = db.get_player_trajectory(p["id"])
        result[str(p["fixed_id"])] = {
            "team": p["team"],
            "points": [[r["x_cm"], r["y_cm"]] for r in traj],
        }

    ball = db.get_ball_trajectory(match_id)
    result["ball"] = {"points": [[r["x_cm"], r["y_cm"]] for r in ball]}
    return JSONResponse(result)
