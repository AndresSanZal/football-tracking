"""database.py
Gestión de la base de datos SQLite.
Un solo fichero web/football.db — cero configuración.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "football.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # permite lecturas concurrentes
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Crea las tablas si no existen."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS match (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            uploaded_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            status        TEXT    NOT NULL DEFAULT 'pending',
            error_msg     TEXT,
            fps           REAL,
            total_frames  INTEGER,
            video_out     TEXT
        );

        CREATE TABLE IF NOT EXISTS player (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id      INTEGER NOT NULL REFERENCES match(id),
            fixed_id      INTEGER NOT NULL,
            team          INTEGER NOT NULL,
            distance_m    REAL,
            avg_speed_kmh REAL,
            max_speed_kmh REAL
        );

        CREATE TABLE IF NOT EXISTS player_trajectory (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL REFERENCES player(id),
            frame     INTEGER NOT NULL,
            x_cm      REAL    NOT NULL,
            y_cm      REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ball_trajectory (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL REFERENCES match(id),
            frame    INTEGER NOT NULL,
            x_cm     REAL    NOT NULL,
            y_cm     REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS possession (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id      INTEGER NOT NULL REFERENCES match(id),
            team0_frames  INTEGER,
            team1_frames  INTEGER,
            team0_pct     REAL,
            team1_pct     REAL
        );
        """)


# ── Matches ────────────────────────────────────────────────────────────────────

def create_match(name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO match (name, status) VALUES (?, 'pending')", (name,)
        )
        return cur.lastrowid


def set_match_processing(match_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE match SET status='processing' WHERE id=?", (match_id,))


def set_match_done(match_id: int, fps: float, total_frames: int, video_out: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE match SET status='done', fps=?, total_frames=?, video_out=? WHERE id=?",
            (fps, total_frames, video_out, match_id),
        )


def set_match_error(match_id: int, msg: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE match SET status='error', error_msg=? WHERE id=?",
            (msg, match_id),
        )


def get_match(match_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM match WHERE id=?", (match_id,)).fetchone()


def list_matches():
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM match ORDER BY uploaded_at DESC"
        ).fetchall()


# ── Importar métricas desde el JSON generado por MetricsExporter ──────────────

def import_metrics(match_id: int, json_path: str) -> None:
    """Lee el JSON de métricas y lo inserta en la BD."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    with get_conn() as conn:
        # Posesión
        poss = data.get("possession", {})
        conn.execute(
            """INSERT INTO possession
               (match_id, team0_frames, team1_frames, team0_pct, team1_pct)
               VALUES (?,?,?,?,?)""",
            (
                match_id,
                poss.get("team0_frames", 0),
                poss.get("team1_frames", 0),
                poss.get("team0_pct", 0.0),
                poss.get("team1_pct", 0.0),
            ),
        )

        # Jugadores
        for pid_str, pdata in data.get("players", {}).items():
            cur = conn.execute(
                """INSERT INTO player
                   (match_id, fixed_id, team, distance_m, avg_speed_kmh, max_speed_kmh)
                   VALUES (?,?,?,?,?,?)""",
                (
                    match_id,
                    int(pid_str),
                    pdata.get("team", -1),
                    pdata.get("distance_m"),
                    pdata.get("avg_speed_kmh"),
                    pdata.get("max_speed_kmh"),
                ),
            )
            player_db_id = cur.lastrowid

            # Trayectoria diezmada a 1 de cada 6 frames para aligerar la BD
            traj = pdata.get("trajectory", [])
            rows = [
                (player_db_id, int(f), float(x), float(y))
                for f, x, y in traj[::6]
            ]
            conn.executemany(
                "INSERT INTO player_trajectory (player_id, frame, x_cm, y_cm) VALUES (?,?,?,?)",
                rows,
            )

        # Trayectoria del balón diezmada igual
        ball_traj = data.get("ball", {}).get("trajectory", [])
        rows = [
            (match_id, int(f), float(x), float(y))
            for f, x, y in ball_traj[::6]
        ]
        conn.executemany(
            "INSERT INTO ball_trajectory (match_id, frame, x_cm, y_cm) VALUES (?,?,?,?)",
            rows,
        )


# ── Consultas para el dashboard ───────────────────────────────────────────────

def get_players(match_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM player WHERE match_id=? ORDER BY fixed_id",
            (match_id,),
        ).fetchall()


def get_possession(match_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM possession WHERE match_id=?", (match_id,)
        ).fetchone()


def get_player_trajectory(player_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT frame, x_cm, y_cm FROM player_trajectory WHERE player_id=? ORDER BY frame",
            (player_id,),
        ).fetchall()


def get_ball_trajectory(match_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT frame, x_cm, y_cm FROM ball_trajectory WHERE match_id=? ORDER BY frame",
            (match_id,),
        ).fetchall()
