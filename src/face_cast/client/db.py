"""SQLite DAO — 装载 schema, 提供 INSERT/SELECT 包装.

设计:
  - 一个连接走全程, 上层不直接写 SQL
  - WAL + foreign_keys ON
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema.sql"


# ─── DB connection / setup ────────────────────────────────────────────────


def connect(db_path: Path | str) -> sqlite3.Connection:
    """打开 SQLite 连接, 应用 schema (idempotent)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.row_factory = sqlite3.Row

    if SCHEMA_PATH.exists():
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


# ─── frame ────────────────────────────────────────────────────────────────


def upsert_frame(
    conn: sqlite3.Connection,
    video_path: str,
    frame_ms: int,
    width: int | None = None,
    height: int | None = None,
) -> int:
    """返回 frame_id (新插或既有)."""
    row = conn.execute(
        "SELECT id FROM frames WHERE video_path = ? AND frame_ms = ?",
        (video_path, frame_ms),
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO frames (video_path, frame_ms, width, height) VALUES (?,?,?,?)",
        (video_path, frame_ms, width, height),
    )
    return cur.lastrowid


def frame_already_processed(
    conn: sqlite3.Connection, video_path: str, frame_ms: int
) -> bool:
    """该帧是否已抽过且至少有过一次推理 (有 face 行)."""
    row = conn.execute(
        """
        SELECT 1 FROM frames f
        JOIN faces fa ON fa.frame_id = f.id
        WHERE f.video_path = ? AND f.frame_ms = ?
        LIMIT 1
        """,
        (video_path, frame_ms),
    ).fetchone()
    return row is not None


# ─── face ─────────────────────────────────────────────────────────────────


def insert_face(
    conn: sqlite3.Connection,
    frame_id: int,
    bbox: tuple[int, int, int, int],
    det_score: float | None,
    detector: str,
    age: int | None,
    sex: int | None,
    crop_jpeg: bytes | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO faces
            (frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             det_score, detector, age, sex, crop_jpeg)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (frame_id, *bbox, det_score, detector, age, sex, crop_jpeg),
    )
    return cur.lastrowid


# ─── embedding ────────────────────────────────────────────────────────────


def insert_embedding(
    conn: sqlite3.Connection,
    face_id: int,
    model_name: str,
    model_version: str,
    vector: np.ndarray,
) -> int:
    """vector 必须是 float32 一维 numpy."""
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    if vector.ndim != 1:
        raise ValueError(f"embedding must be 1-D, got {vector.shape}")
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO embeddings
            (face_id, model_name, model_version, dim, vector)
        VALUES (?,?,?,?,?)
        """,
        (face_id, model_name, model_version, len(vector), vector.tobytes()),
    )
    return cur.lastrowid or 0


def load_embeddings(
    conn: sqlite3.Connection, model_name: str, model_version: str | None = None
) -> tuple[list[int], np.ndarray]:
    """加载某模型的全部 embedding, 返回 (face_ids, matrix shape=(N, dim))."""
    if model_version:
        rows = conn.execute(
            """
            SELECT face_id, dim, vector FROM embeddings
            WHERE model_name = ? AND model_version = ?
            ORDER BY face_id
            """,
            (model_name, model_version),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT face_id, dim, vector FROM embeddings
            WHERE model_name = ?
            ORDER BY face_id
            """,
            (model_name,),
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = rows[0]["dim"]
    face_ids = [r["face_id"] for r in rows]
    matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(-1, dim)
    return face_ids, matrix


# ─── detection_runs / persons / face_samples ─────────────────────────────


def create_run(
    conn: sqlite3.Connection,
    model_name: str,
    model_version: str,
    algo: str,
    params: dict,
    n_embeddings: int,
    notes: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO detection_runs
            (model_name, model_version, algo, params_json, n_embeddings, notes)
        VALUES (?,?,?,?,?,?)
        """,
        (model_name, model_version, algo, json.dumps(params, ensure_ascii=False),
         n_embeddings, notes),
    )
    return cur.lastrowid


def finalize_run(
    conn: sqlite3.Connection,
    run_id: int,
    n_persons: int,
    n_noise: int,
    set_active: bool = True,
) -> None:
    if set_active:
        conn.execute("UPDATE detection_runs SET is_active = 0")
    conn.execute(
        "UPDATE detection_runs SET n_persons = ?, n_noise = ?, is_active = ? WHERE id = ?",
        (n_persons, n_noise, 1 if set_active else 0, run_id),
    )


def insert_person(
    conn: sqlite3.Connection,
    run_id: int,
    person_idx: int,
    size: int,
    centroid: np.ndarray | None = None,
) -> int:
    blob = centroid.astype(np.float32).tobytes() if centroid is not None else None
    cur = conn.execute(
        """
        INSERT INTO persons (run_id, person_idx, size, centroid_blob)
        VALUES (?,?,?,?)
        """,
        (run_id, person_idx, size, blob),
    )
    return cur.lastrowid


def link_face_sample(
    conn: sqlite3.Connection, face_id: int, person_id: int, is_manual: bool = False
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO face_samples (face_id, person_id, is_manual) VALUES (?,?,?)",
        (face_id, person_id, 1 if is_manual else 0),
    )


def active_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM detection_runs WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
    ).fetchone()


# ─── 给 NFO 写入用: 查每个 video 的 person ────────────────────────────────


def video_persons(
    conn: sqlite3.Connection, video_path: str, run_id: int, min_appearances: int = 2
) -> list[sqlite3.Row]:
    """返回某 video 在该 run 下的主要 person (出场次数 >= 阈值, 排除噪声 -1)."""
    return list(
        conn.execute(
            """
            SELECT
                p.id AS person_id,
                p.person_idx,
                p.display_name,
                p.size AS person_size,
                COUNT(fs.face_id) AS appearances
            FROM frames f
            JOIN faces fa ON fa.frame_id = f.id
            JOIN face_samples fs ON fs.face_id = fa.id
            JOIN persons p ON p.id = fs.person_id
            WHERE f.video_path = ?
              AND p.run_id = ?
              AND p.person_idx >= 0          -- 排除噪声 -1
            GROUP BY p.id
            HAVING appearances >= ?
            ORDER BY appearances DESC
            """,
            (video_path, run_id, min_appearances),
        ).fetchall()
    )


# ─── 工作日志 ────────────────────────────────────────────────────────────


def log(
    conn: sqlite3.Connection,
    op: str,
    target: str | None,
    status: str,
    detail: str | None = None,
    duration_ms: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO work_log (op, target, status, detail, duration_ms)
        VALUES (?,?,?,?,?)
        """,
        (op, target, status, detail, duration_ms),
    )
