"""SQLite DAO — 装载 schema, 提供 INSERT/SELECT 包装.

设计:
  - 一个连接走全程, 上层不直接写 SQL
  - 用 Pydantic / dataclass 暴露行类型 (这里用 dataclass 减依赖)
  - WAL + foreign_keys ON
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema.sql"


# ─── row dataclasses ───────────────────────────────────────────────────────


@dataclass
class FrameRow:
    id: int
    video_path: str
    frame_ms: int
    width: int | None
    height: int | None


@dataclass
class FaceRow:
    id: int
    frame_id: int
    bbox: tuple[int, int, int, int]
    det_score: float | None
    detector: str
    age: int | None
    sex: int | None


@dataclass
class EmbeddingRow:
    id: int
    face_id: int
    model_name: str
    model_version: str
    dim: int
    vector: np.ndarray


# ─── DB connection / setup ────────────────────────────────────────────────


def connect(db_path: Path | str) -> sqlite3.Connection:
    """打开 SQLite 连接, 应用 schema (idempotent)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit, 显式 BEGIN
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
    """该帧是否已抽过且至少有过一次推理 (有 face 行 OR 有 work_log embed ok)."""
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


def faces_by_frame(conn: sqlite3.Connection, frame_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute("SELECT * FROM faces WHERE frame_id = ?", (frame_id,)).fetchall()
    )


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


# ─── cluster_runs / clusters / face_cluster ──────────────────────────────


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
        INSERT INTO cluster_runs
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
    n_clusters: int,
    n_noise: int,
    set_active: bool = True,
) -> None:
    if set_active:
        conn.execute("UPDATE cluster_runs SET is_active = 0")
    conn.execute(
        "UPDATE cluster_runs SET n_clusters = ?, n_noise = ?, is_active = ? WHERE id = ?",
        (n_clusters, n_noise, 1 if set_active else 0, run_id),
    )


def insert_cluster(
    conn: sqlite3.Connection,
    run_id: int,
    cluster_idx: int,
    size: int,
    centroid: np.ndarray | None = None,
) -> int:
    blob = centroid.astype(np.float32).tobytes() if centroid is not None else None
    cur = conn.execute(
        """
        INSERT INTO clusters (run_id, cluster_idx, size, centroid_blob)
        VALUES (?,?,?,?)
        """,
        (run_id, cluster_idx, size, blob),
    )
    return cur.lastrowid


def link_face_cluster(conn: sqlite3.Connection, face_id: int, cluster_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO face_cluster (face_id, cluster_id) VALUES (?,?)",
        (face_id, cluster_id),
    )


def active_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM cluster_runs WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
    ).fetchone()


# ─── 给 NFO 写入用: 查每个 video 的 actor cluster ────────────────────────


def video_actors(
    conn: sqlite3.Connection, video_path: str, run_id: int, min_appearances: int = 2
) -> list[sqlite3.Row]:
    """返回某 video 在该 run 下的主要 cluster (出场次数 >= 阈值, 排除噪声)."""
    return list(
        conn.execute(
            """
            SELECT
                c.id AS cluster_id,
                c.cluster_idx,
                c.human_name,
                c.size AS cluster_size,
                COUNT(fc.face_id) AS appearances
            FROM frames f
            JOIN faces fa ON fa.frame_id = f.id
            JOIN face_cluster fc ON fc.face_id = fa.id
            JOIN clusters c ON c.id = fc.cluster_id
            WHERE f.video_path = ?
              AND c.run_id = ?
              AND c.cluster_idx >= 0          -- 排除噪声 -1
            GROUP BY c.id
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
