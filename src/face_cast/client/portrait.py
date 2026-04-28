"""导出代表头像 — 给每个 named person 选一张 face crop, 写到 .actors/.

依赖:
  - faces.crop_jpeg 必须有 (跑 phase2 时没用 --no-crop-cache)
  - persons.centroid_blob 用作"典型脸"参考点

挑选公式:
  score = det_score × (1 - cosine_dist_to_centroid)
  取最高分的 face, 它的 crop_jpeg 当头像

Jellyfin 行为: 看到 NFO 里 <actor><name>X</name>, 自动扫该视频同目录
.actors/X.jpg 当头像, 全局缓存. 不需要在 NFO 里写 <thumb>.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

console = Console()

# Windows 文件名非法字符 + 一些问题字符
_BAD_FN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class PortraitPlan:
    person_id: int
    person_idx: int
    display_name: str
    safe_name: str             # 文件系统安全的名字
    thumb_jpeg: bytes
    target_dirs: list[Path]    # .actors/<safe_name>.jpg 要写到哪些视频目录


def _safe_filename(name: str) -> str:
    """剥掉 Windows 非法字符 + 折叠空白. 截到 80 字符."""
    s = _BAD_FN_CHARS.sub("", name)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80] or "unnamed"


def pick_representative(
    conn: sqlite3.Connection, person_db_id: int
) -> bytes | None:
    """对单个 person 选代表 face crop. 没有可用 crop 时返回 None."""
    # 1. 拿 centroid
    row = conn.execute(
        "SELECT centroid_blob FROM persons WHERE id = ?", (person_db_id,)
    ).fetchone()
    if row is None or row["centroid_blob"] is None:
        return None
    centroid = np.frombuffer(row["centroid_blob"], dtype=np.float32)
    # L2 normalize centroid (cosine 距离需要)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return None
    centroid_n = centroid / norm

    # 2. JOIN 拿该 person 全部 (face_id, det_score, embedding, crop_jpeg)
    rows = conn.execute(
        """
        SELECT f.id AS face_id,
               f.det_score,
               f.crop_jpeg,
               e.vector,
               e.dim
        FROM face_samples fs
        JOIN faces f ON f.id = fs.face_id
        JOIN embeddings e ON e.face_id = f.id
        JOIN persons p ON p.id = fs.person_id
        WHERE fs.person_id = ?
          AND f.crop_jpeg IS NOT NULL          -- 必须有缓存的 crop
          AND e.model_name = (SELECT model_name FROM detection_runs WHERE id = p.run_id)
          AND e.model_version = (SELECT model_version FROM detection_runs WHERE id = p.run_id)
        """,
        (person_db_id,),
    ).fetchall()
    if not rows:
        return None

    # 3. 算每张脸的 score, 取最高
    best_score = -1.0
    best_jpg: bytes | None = None
    for r in rows:
        emb = np.frombuffer(r["vector"], dtype=np.float32)
        n = np.linalg.norm(emb)
        if n == 0:
            continue
        emb_n = emb / n
        # cosine_similarity ∈ [-1, 1], 1 = identical
        sim = float(np.dot(emb_n, centroid_n))
        # score: det_score * sim, sim 越接近 1 越像 centroid
        score = (r["det_score"] or 0.0) * sim
        if score > best_score:
            best_score = score
            best_jpg = r["crop_jpeg"]

    return best_jpg


def _person_video_dirs(
    conn: sqlite3.Connection, person_db_id: int
) -> list[Path]:
    """该 person 出现过的所有视频文件所在目录 (去重)."""
    rows = conn.execute(
        """
        SELECT DISTINCT f.video_path
        FROM face_samples fs
        JOIN faces fa ON fa.id = fs.face_id
        JOIN frames f ON f.id = fa.frame_id
        WHERE fs.person_id = ?
        """,
        (person_db_id,),
    ).fetchall()
    return list({Path(r["video_path"]).parent for r in rows})


def plan_exports(
    conn: sqlite3.Connection,
    run_id: int,
    include_unnamed: bool = False,
) -> Iterable[PortraitPlan]:
    """生成每个 person 的导出计划."""
    where = "WHERE run_id = ? AND person_idx >= 0"
    params: tuple = (run_id,)
    if not include_unnamed:
        where += " AND display_name IS NOT NULL"

    persons = conn.execute(
        f"SELECT id, person_idx, display_name FROM persons {where}",
        params,
    ).fetchall()

    for p in persons:
        thumb = pick_representative(conn, p["id"])
        if thumb is None:
            continue
        display_name = p["display_name"] or f"person_{p['person_idx']}"
        safe = _safe_filename(display_name)
        dirs = _person_video_dirs(conn, p["id"])
        if not dirs:
            continue
        yield PortraitPlan(
            person_id=p["id"],
            person_idx=p["person_idx"],
            display_name=display_name,
            safe_name=safe,
            thumb_jpeg=thumb,
            target_dirs=dirs,
        )


def export(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    include_unnamed: bool = False,
    redundant: bool = True,
    dry_run: bool = False,
) -> dict:
    """对每个 named person 写 .actors/<safe_name>.jpg 到目标目录.

    redundant=True (默认): 每个相关视频目录都写一份 (~3 KB × N 视频, 总量小)
    redundant=False: 只在该 person 出现的"第一个"目录写一份, 靠 Jellyfin 全局缓存
                     (风险: 如果那个视频被删, 头像也丢)
    """
    plans = list(plan_exports(conn, run_id, include_unnamed=include_unnamed))
    if not plans:
        console.print("[yellow]没有可导出的 person (是否跑过 detect + name-person?)[/yellow]")
        return {"persons": 0, "files": 0, "dirs": 0}

    n_files = 0
    n_dirs = 0
    seen_dirs: set[Path] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as pg:
        task = pg.add_task("persons", total=len(plans))
        for plan in plans:
            dirs = plan.target_dirs if redundant else plan.target_dirs[:1]
            pg.update(task, description=f"{plan.display_name} ({len(dirs)} dirs)")
            for d in dirs:
                actors = d / ".actors"
                target = actors / f"{plan.safe_name}.jpg"
                if dry_run:
                    console.print(f"  [dim]would write {target} ({len(plan.thumb_jpeg)} B)[/dim]")
                else:
                    actors.mkdir(exist_ok=True)
                    target.write_bytes(plan.thumb_jpeg)
                if d not in seen_dirs:
                    seen_dirs.add(d)
                    n_dirs += 1
                n_files += 1
            pg.advance(task)

    return {"persons": len(plans), "files": n_files, "dirs": n_dirs}
