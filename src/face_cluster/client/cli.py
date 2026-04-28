"""``face-cluster`` 客户端 CLI — 走 typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .api import FaceClient
from .phase2 import (
    Phase2Config,
    extract_and_embed,
    run_clustering,
    run_full,
    scan_videos,
    write_nfos,
)

app = typer.Typer(
    help="face-cluster 客户端 — 抽帧 + 调用 server + 聚类 + 写 NFO",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


# ─── 选项 ────────────────────────────────────────────────────────────────

ServerOpt = Annotated[
    str, typer.Option("--server", "-s", help="face-server 地址", envvar="FACE_SERVER_URL")
]
DBOpt = Annotated[
    Path, typer.Option("--db", help="SQLite 工作 DB 路径", envvar="FACE_DB_PATH")
]


# ─── 子命令 ──────────────────────────────────────────────────────────────


@app.command()
def health(server: ServerOpt = "http://10.100.100.11:9000"):
    """探活: 看 server 还在不在."""
    client = FaceClient(server)
    console.print_json(data=client.health())
    info = client.model_info()
    console.print(f"[green]✓[/green] 模型 [bold]{info.model_name}[/bold] v{info.model_version}, dim={info.embedding_dim}")


@app.command()
def run(
    media: Annotated[Path, typer.Argument(help="媒体根目录, 如 F:\\China")],
    server: ServerOpt = "http://10.100.100.11:9000",
    db_path: DBOpt = Path("./face_cluster.db"),
    frames: Annotated[int, typer.Option("--frames", "-n", help="每视频抽几帧")] = 5,
    no_crop_cache: Annotated[bool, typer.Option("--no-crop-cache", help="不缓存 face_jpeg (省空间, 牺牲未来重 embed 速度)")] = False,
):
    """端到端 Phase 2: scan → extract+embed → cluster → write NFO."""
    cfg = Phase2Config(
        media_root=media.resolve(),
        db_path=db_path.resolve(),
        server_url=server,
        frames_per_video=frames,
        cache_face_crops=not no_crop_cache,
    )
    run_full(cfg)


@app.command()
def extract(
    media: Annotated[Path, typer.Argument(help="媒体根目录")],
    server: ServerOpt = "http://10.100.100.11:9000",
    db_path: DBOpt = Path("./face_cluster.db"),
    frames: Annotated[int, typer.Option("--frames", "-n")] = 5,
    no_crop_cache: bool = False,
):
    """只跑抽帧 + embedding, 不聚类."""
    cfg = Phase2Config(
        media_root=media.resolve(),
        db_path=db_path.resolve(),
        server_url=server,
        frames_per_video=frames,
        cache_face_crops=not no_crop_cache,
    )
    conn = db.connect(cfg.db_path)
    client = FaceClient(cfg.server_url)
    info = client.model_info()
    videos = scan_videos(cfg.media_root)
    console.print(f"[cyan]发现 {len(videos):,} 个视频[/cyan]")
    extract_and_embed(cfg, conn, client, info, videos)


@app.command()
def cluster(
    server: ServerOpt = "http://10.100.100.11:9000",
    db_path: DBOpt = Path("./face_cluster.db"),
    min_cluster_size: int = 3,
    min_samples: int = 2,
):
    """只跑 HDBSCAN 聚类 (DB 里已有 embedding 时用)."""
    cfg = Phase2Config(
        media_root=Path("."),
        db_path=db_path.resolve(),
        server_url=server,
        hdbscan_min_cluster=min_cluster_size,
        hdbscan_min_samples=min_samples,
    )
    conn = db.connect(cfg.db_path)
    client = FaceClient(cfg.server_url)
    info = client.model_info()
    run_clustering(cfg, conn, info)


@app.command(name="write-nfo")
def cmd_write_nfo(
    db_path: DBOpt = Path("./face_cluster.db"),
    run_id: Annotated[int | None, typer.Option(help="使用哪个 cluster_run, 默认当前 active")] = None,
    min_appearances: int = 2,
):
    """只写 NFO (聚类已完成时用, 比如手动改了 human_name 后重写)."""
    cfg = Phase2Config(
        media_root=Path("."),
        db_path=db_path.resolve(),
        server_url="",
        nfo_min_appearances=min_appearances,
    )
    conn = db.connect(cfg.db_path)
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run, 跑一次 cluster 先 (or 给 --run-id)")
        run_id = active["id"]
    write_nfos(cfg, conn, run_id)


@app.command(name="list-clusters")
def cmd_list_clusters(
    db_path: DBOpt = Path("./face_cluster.db"),
    run_id: Annotated[int | None, typer.Option()] = None,
    limit: int = 50,
):
    """列出某 run 下所有 cluster (默认当前 active), 按 size 排序."""
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            console.print("[red]没有 active run[/red]")
            raise typer.Exit(1)
        run_id = active["id"]

    rows = list(conn.execute(
        """
        SELECT cluster_idx, size, human_name
        FROM clusters
        WHERE run_id = ?
        ORDER BY size DESC
        LIMIT ?
        """,
        (run_id, limit),
    ))

    table = Table(title=f"clusters · run #{run_id}", show_header=True)
    table.add_column("cluster_idx", justify="right")
    table.add_column("size", justify="right")
    table.add_column("human_name")
    for r in rows:
        table.add_row(
            str(r["cluster_idx"]),
            str(r["size"]),
            r["human_name"] or "[dim](unnamed)[/dim]",
        )
    console.print(table)


@app.command(name="name-cluster")
def cmd_name_cluster(
    cluster_idx: Annotated[int, typer.Argument(help="cluster_idx 值")],
    name: Annotated[str, typer.Argument(help="人类名字")],
    db_path: DBOpt = Path("./face_cluster.db"),
    run_id: Annotated[int | None, typer.Option()] = None,
):
    """给某个 cluster 起名: face-cluster name-cluster 7 '李雅'."""
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run")
        run_id = active["id"]
    cur = conn.execute(
        "UPDATE clusters SET human_name = ? WHERE run_id = ? AND cluster_idx = ?",
        (name, run_id, cluster_idx),
    )
    if cur.rowcount == 0:
        console.print(f"[red]cluster_idx={cluster_idx} 不存在 (run #{run_id})[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] cluster {cluster_idx} → '{name}'")
    console.print(f"[dim]改完后 face-cluster write-nfo 重写 NFO[/dim]")


@app.command(name="stats")
def cmd_stats(db_path: DBOpt = Path("./face_cluster.db")):
    """整体统计: 视频/帧/脸/embedding/cluster 数."""
    conn = db.connect(db_path.resolve())
    rows = {
        "videos": "SELECT COUNT(DISTINCT video_path) FROM frames",
        "frames": "SELECT COUNT(*) FROM frames",
        "faces": "SELECT COUNT(*) FROM faces",
        "embeddings": "SELECT COUNT(*) FROM embeddings",
        "models": "SELECT COUNT(DISTINCT model_name||model_version) FROM embeddings",
        "cluster_runs": "SELECT COUNT(*) FROM cluster_runs",
        "active_run": "SELECT id FROM cluster_runs WHERE is_active=1 LIMIT 1",
    }
    table = Table(show_header=False)
    for k, q in rows.items():
        v = conn.execute(q).fetchone()
        table.add_row(k, str(v[0] if v else "—"))
    console.print(table)


def cli() -> None:
    """``face-cluster`` 入口 (pyproject 注册)."""
    app()


if __name__ == "__main__":
    cli()
