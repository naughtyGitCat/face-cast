"""主编排 — 把所有模块串起来.

流程:
  scan_videos        → 找视频
  extract_and_embed  → 抽帧 + POST 服务端 + 写 SQLite (frames/faces/embeddings)
  detect_persons     → HDBSCAN 跑识别, 写 detection_runs/persons/face_samples
  write_nfos         → 给每个 video 改 NFO, 加 actor

每步都幂等 (查 SQLite 跳过), 中断后可断点续跑.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import db
from .api import FaceClient, ServerInfo
from .cluster import detect_persons
from .extract import crop_face, extract_frame, ffprobe, sample_timestamps
from .nfo import ActorInfo, nfo_path_for, update_actors

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v", ".mpg", ".webm", ".flv", ".rmvb", ".wmv"}

console = Console()
err_console = Console(stderr=True)


@dataclass
class Config:
    media_root: Path
    db_path: Path
    server_url: str
    frames_per_video: int = 5
    edge_margin: float = 0.1                # 跳过首尾各 10%
    max_frame_height: int = 720             # 抽帧时 ffmpeg 端下采样
    cache_face_crops: bool = True           # 写 faces.crop_jpeg
    crop_size: int = 224
    hdbscan_min_cluster: int = 3
    hdbscan_min_samples: int = 2
    nfo_min_appearances: int = 2            # 至少 N 帧出现才认为是 actor


# ─── 1. 扫描视频 ──────────────────────────────────────────────────────────


def scan_videos(media_root: Path) -> list[Path]:
    return sorted(
        p for p in media_root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS and p.is_file()
    )


# ─── 2. 抽帧 + 推理 ──────────────────────────────────────────────────────


def extract_and_embed(
    cfg: Config,
    conn: sqlite3.Connection,
    client: FaceClient,
    server: ServerInfo,
    videos: Iterable[Path],
) -> None:
    videos = list(videos)
    detector = server.detector

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("视频", total=len(videos))
        for video in videos:
            progress.update(task, description=video.name[:40])
            _process_one_video(cfg, conn, client, server, detector, video)
            progress.advance(task)


def _process_one_video(
    cfg: Config,
    conn: sqlite3.Connection,
    client: FaceClient,
    server: ServerInfo,
    detector: str,
    video: Path,
) -> None:
    rel = str(video)
    meta = ffprobe(video)
    if meta is None:
        db.log(conn, "extract", rel, "fail", "ffprobe failed")
        return

    timestamps = sample_timestamps(meta.duration_s, cfg.frames_per_video, cfg.edge_margin)
    if not timestamps:
        db.log(conn, "extract", rel, "skip", "duration<=0")
        return

    for ms in timestamps:
        if db.frame_already_processed(conn, rel, ms):
            continue
        t0 = time.time()
        jpg = extract_frame(video, ms, max_height=cfg.max_frame_height)
        if jpg is None:
            db.log(conn, "extract", f"{rel}@{ms}", "fail", "ffmpeg extract failed",
                   int((time.time() - t0) * 1000))
            continue

        # POST 单帧, 拿 faces
        try:
            faces = client.detect(jpg, filename=f"{video.stem}_{ms}.jpg")
        except Exception as e:  # noqa: BLE001
            db.log(conn, "embed", f"{rel}@{ms}", "fail", str(e)[:200])
            continue

        # 写 frame
        frame_id = db.upsert_frame(conn, rel, ms, meta.width, meta.height)

        for face in faces:
            crop_bytes = (
                crop_face(jpg, face.bbox, out_size=cfg.crop_size) if cfg.cache_face_crops else None
            )
            face_id = db.insert_face(
                conn,
                frame_id=frame_id,
                bbox=face.bbox,
                det_score=face.det_score,
                detector=detector,
                age=face.age,
                sex=face.sex,
                crop_jpeg=crop_bytes,
            )
            db.insert_embedding(
                conn,
                face_id=face_id,
                model_name=server.model_name,
                model_version=server.model_version,
                vector=face.embedding,
            )

        db.log(conn, "embed", f"{rel}@{ms}", "ok",
               f"faces={len(faces)}", int((time.time() - t0) * 1000))


# ─── 3. 识别 person ──────────────────────────────────────────────────────


def run_detection(
    cfg: Config,
    conn: sqlite3.Connection,
    server: ServerInfo,
) -> int:
    """跑 HDBSCAN 识别 person, 写库. 返回 run_id."""
    console.print("[cyan]载入 embeddings...[/cyan]")
    face_ids, matrix = db.load_embeddings(conn, server.model_name, server.model_version)
    if matrix.shape[0] == 0:
        raise RuntimeError("no embeddings yet, run extract first")
    console.print(f"  {matrix.shape[0]:,} 个 embedding ({matrix.shape[1]} dim)")

    params = {
        "min_cluster_size": cfg.hdbscan_min_cluster,
        "min_samples": cfg.hdbscan_min_samples,
        "metric": "cosine",
    }
    run_id = db.create_run(
        conn,
        model_name=server.model_name,
        model_version=server.model_version,
        algo="hdbscan",
        params=params,
        n_embeddings=matrix.shape[0],
    )

    console.print("[cyan]HDBSCAN 跑起来...[/cyan]")
    t0 = time.time()
    result = detect_persons(matrix, **params)
    console.print(
        f"  {result.n_persons} 个 person · {result.n_noise} 噪声 · "
        f"耗时 {time.time() - t0:.1f}s"
    )

    # 写 persons + face_samples
    person_id_map: dict[int, int] = {}
    for person_idx, centroid in result.centroids.items():
        size = int((result.labels == person_idx).sum())
        person_db_id = db.insert_person(conn, run_id, person_idx, size, centroid)
        person_id_map[person_idx] = person_db_id

    # 噪声单独存一行 person_idx=-1, 方便 UI 区分但不参与 actor 写入
    if result.n_noise > 0:
        person_id_map[-1] = db.insert_person(conn, run_id, -1, result.n_noise, None)

    for face_id, label in zip(face_ids, result.labels, strict=True):
        pid = person_id_map.get(int(label))
        if pid is not None:
            db.link_face_sample(conn, face_id, pid)

    db.finalize_run(conn, run_id, result.n_persons, result.n_noise, set_active=True)
    db.log(conn, "detect", str(run_id), "ok",
           f"persons={result.n_persons} noise={result.n_noise}")
    console.print(f"[green]✓ detection run #{run_id} 已激活[/green]")
    return run_id


# ─── 4. 写 NFO ───────────────────────────────────────────────────────────


def write_nfos(
    cfg: Config,
    conn: sqlite3.Connection,
    run_id: int,
    videos: Iterable[Path] | None = None,
) -> None:
    if videos is None:
        videos = [Path(r["video_path"]) for r in
                  conn.execute("SELECT DISTINCT video_path FROM frames")]
    videos = list(videos)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("NFO", total=len(videos))
        for video in videos:
            progress.update(task, description=video.name[:40])
            persons = db.video_persons(conn, str(video), run_id, cfg.nfo_min_appearances)
            actor_infos = [
                ActorInfo(name=row["display_name"] or f"person_{row['person_idx']}")
                for row in persons
            ]
            nfo = nfo_path_for(video)
            ok = update_actors(nfo, actor_infos, create_if_missing=False)
            if ok:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO nfo_state
                        (video_path, nfo_path, actors_json, last_run_id, last_written_at)
                    VALUES (?,?,?,?, CURRENT_TIMESTAMP)
                    """,
                    (str(video), str(nfo),
                     json.dumps([a.name for a in actor_infos], ensure_ascii=False),
                     run_id),
                )
                db.log(conn, "nfo", str(video), "ok", f"{len(actor_infos)} actors")
            else:
                db.log(conn, "nfo", str(video), "skip", "no nfo file")
            progress.advance(task)


# ─── 整合 ─────────────────────────────────────────────────────────────────


def run_full(cfg: Config) -> None:
    """端到端跑一遍."""
    conn = db.connect(cfg.db_path)
    client = FaceClient(cfg.server_url)

    console.print(f"[cyan]health check {cfg.server_url}...[/cyan]")
    h = client.health()
    if not h.get("ok"):
        raise RuntimeError(f"server not ready: {h}")
    server = client.model_info()
    console.print(
        f"  ✓ 模型 [bold]{server.model_name}[/bold] v{server.model_version} · "
        f"dim={server.embedding_dim}"
    )

    console.print(f"[cyan]扫描 {cfg.media_root}...[/cyan]")
    videos = scan_videos(cfg.media_root)
    console.print(f"  发现 {len(videos):,} 个视频")
    if not videos:
        return

    extract_and_embed(cfg, conn, client, server, videos)
    run_id = run_detection(cfg, conn, server)
    write_nfos(cfg, conn, run_id, videos)

    console.print(f"[bold green]done[/bold green] · run_id={run_id}")
