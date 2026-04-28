"""``face-cast`` 客户端 CLI — 走 typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import db, jellyfin, portrait
from .api import FaceClient
from .cluster import split_subcluster
from .phase2 import (
    Config,
    extract_and_embed,
    run_detection,
    run_full,
    scan_videos,
    write_nfos,
)

app = typer.Typer(
    help="face-cast 客户端 — 抽帧 + 调用 server + 识别 person + 写 NFO",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


# ─── 选项 ────────────────────────────────────────────────────────────────

ServerOpt = Annotated[
    str, typer.Option("--server", "-s", help="face-cast server 地址", envvar="FACE_SERVER_URL")
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
    console.print(
        f"[green]✓[/green] 模型 [bold]{info.model_name}[/bold] "
        f"v{info.model_version}, dim={info.embedding_dim}"
    )


@app.command()
def run(
    media: Annotated[Path, typer.Argument(help="媒体根目录, 如 F:\\China")],
    server: ServerOpt = "http://10.100.100.11:9000",
    db_path: DBOpt = Path("./face_cast.db"),
    frames: Annotated[int, typer.Option("--frames", "-n", help="每视频抽几帧")] = 15,
    no_crop_cache: Annotated[
        bool, typer.Option("--no-crop-cache", help="不缓存 face crop (省空间, 牺牲未来重 embed 速度)")
    ] = False,
):
    """端到端: scan → extract+embed → detect → write NFO."""
    cfg = Config(
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
    db_path: DBOpt = Path("./face_cast.db"),
    frames: Annotated[int, typer.Option("--frames", "-n")] = 15,
    no_crop_cache: bool = False,
):
    """只跑抽帧 + embedding, 不做 person 识别."""
    cfg = Config(
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
def detect(
    server: ServerOpt = "http://10.100.100.11:9000",
    db_path: DBOpt = Path("./face_cast.db"),
    min_cluster_size: int = 3,
    min_samples: int = 2,
):
    """只跑 HDBSCAN 识别 person (DB 里已有 embedding 时用)."""
    cfg = Config(
        media_root=Path("."),
        db_path=db_path.resolve(),
        server_url=server,
        hdbscan_min_cluster=min_cluster_size,
        hdbscan_min_samples=min_samples,
    )
    conn = db.connect(cfg.db_path)
    client = FaceClient(cfg.server_url)
    info = client.model_info()
    run_detection(cfg, conn, info)


@app.command(name="write-nfo")
def cmd_write_nfo(
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[
        int | None, typer.Option(help="使用哪个 detection_run, 默认当前 active")
    ] = None,
    min_appearances: int = 2,
):
    """只写 NFO (识别已完成时用, 比如手动改了 display_name 后重写)."""
    cfg = Config(
        media_root=Path("."),
        db_path=db_path.resolve(),
        server_url="",
        nfo_min_appearances=min_appearances,
    )
    conn = db.connect(cfg.db_path)
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run, 跑一次 detect 先 (or 给 --run-id)")
        run_id = active["id"]
    write_nfos(cfg, conn, run_id)


@app.command(name="list-persons")
def cmd_list_persons(
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[int | None, typer.Option()] = None,
    limit: int = 50,
):
    """列出某 run 下所有 person (默认当前 active), 按 size 排序."""
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            console.print("[red]没有 active run[/red]")
            raise typer.Exit(1)
        run_id = active["id"]

    rows = list(conn.execute(
        """
        SELECT person_idx, size, display_name
        FROM persons
        WHERE run_id = ?
        ORDER BY size DESC
        LIMIT ?
        """,
        (run_id, limit),
    ))

    table = Table(title=f"persons · run #{run_id}", show_header=True)
    table.add_column("person_idx", justify="right")
    table.add_column("size", justify="right")
    table.add_column("display_name")
    for r in rows:
        table.add_row(
            str(r["person_idx"]),
            str(r["size"]),
            r["display_name"] or "[dim](unnamed)[/dim]",
        )
    console.print(table)


@app.command(name="name-person")
def cmd_name_person(
    person_idx: Annotated[int, typer.Argument(help="person_idx 值")],
    name: Annotated[str, typer.Argument(help="人类名字")],
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[int | None, typer.Option()] = None,
):
    """给某个 person 起名: face-cast name-person 7 '李雅'."""
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run")
        run_id = active["id"]
    cur = conn.execute(
        "UPDATE persons SET display_name = ? WHERE run_id = ? AND person_idx = ?",
        (name, run_id, person_idx),
    )
    if cur.rowcount == 0:
        console.print(f"[red]person_idx={person_idx} 不存在 (run #{run_id})[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] person {person_idx} → '{name}'")
    console.print("[dim]改完后 face-cast write-nfo 重写 NFO[/dim]")


@app.command(name="export-portraits")
def cmd_export_portraits(
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[int | None, typer.Option(help="使用哪个 detection_run, 默认 active")] = None,
    include_unnamed: Annotated[
        bool, typer.Option("--include-unnamed", help="也导出未命名 person (用 person_N 当文件名)")
    ] = False,
    no_redundant: Annotated[
        bool, typer.Option("--no-redundant", help="只在第一个相关目录放一份, 靠 Jellyfin 全局缓存")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只打印, 不写文件")] = False,
):
    """给每个 person 选代表头像, 写 .actors/<name>.jpg 到视频目录.

    Jellyfin 扫到 NFO 里 <actor><name>X</name></actor> 后, 自动找同目录
    .actors/X.jpg 当头像. 不需要在 NFO 里加 <thumb>.
    """
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run, 跑 detect 先")
        run_id = active["id"]
    stats = portrait.export(
        conn,
        run_id=run_id,
        include_unnamed=include_unnamed,
        redundant=not no_redundant,
        dry_run=dry_run,
    )
    console.print(
        f"\n[bold green]导出完成[/bold green]: "
        f"{stats['persons']} 个 person · {stats['files']} 个文件 · {stats['dirs']} 个目录"
    )
    if dry_run:
        console.print("[dim](dry-run, 实际未写入)[/dim]")


@app.command(name="merge-persons")
def cmd_merge_persons(
    target: Annotated[int, typer.Argument(help="合并到这个 person_idx (target)")],
    sources: Annotated[list[int], typer.Argument(help="源 person_idx 列表 (会被合入 target)")],
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[int | None, typer.Option(help="默认 active run")] = None,
    keep_sources: Annotated[
        bool, typer.Option("--keep-sources", help="保留源 person 行 (size 设 0); 默认删除")
    ] = False,
):
    """合并多个 person 到一个: face-cast merge-persons 2 0 1 3.

    把 person 0/1/3 的所有 face 移到 person 2 (target). 之后 face-cast write-nfo
    会用合并后的结果重写 NFO.
    """
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run")
        run_id = active["id"]
    try:
        result = db.merge_persons(
            conn,
            run_id=run_id,
            target_idx=target,
            source_indices=sources,
            delete_sources=not keep_sources,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(
        f"[green]✓[/green] {len(sources)} 个 person 合入 person {target}: "
        f"{result['moved_face_samples']} 张脸, target 现有 {result['new_target_size']} 张"
    )
    console.print("[dim]改完跑 face-cast write-nfo + push-jellyfin 同步到外部[/dim]")


@app.command(name="split-person")
def cmd_split_person(
    person_idx: Annotated[int, typer.Argument(help="要拆开的 person_idx")],
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[int | None, typer.Option()] = None,
    min_cluster_size: int = 2,
    min_samples: int = 2,
    dry_run: bool = False,
):
    """把一个 cluster 重新聚类拆成多个 person (适用于把多人误合的情况).

    对该 person 的 embedding 子集跑 HDBSCAN (参数更宽松), 子聚类输出新 person.
    """
    import numpy as np  # noqa: PLC0415

    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run")
        run_id = active["id"]
    pdb = db.person_db_id(conn, run_id, person_idx)
    if pdb is None:
        raise typer.BadParameter(f"person_idx={person_idx} 不存在")

    rows = list(conn.execute(
        """
        SELECT fs.face_id, e.vector, e.dim
        FROM face_samples fs
        JOIN embeddings e ON e.face_id = fs.face_id
        JOIN persons p ON p.id = fs.person_id
        WHERE fs.person_id = ?
          AND e.model_name = (SELECT model_name FROM detection_runs WHERE id = p.run_id)
        """,
        (pdb,),
    ))
    if not rows:
        console.print("[yellow]该 person 无 embedding[/yellow]")
        return
    dim = rows[0]["dim"]
    face_ids = [r["face_id"] for r in rows]
    matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(-1, dim)

    result = split_subcluster(matrix, min_cluster_size, min_samples)
    console.print(
        f"split: {result.n_persons} 个子 cluster · {result.n_noise} 噪声 (从 {len(face_ids)} 张脸)"
    )
    if result.n_persons == 0:
        console.print("[yellow]子聚类未产出 cluster, 不动 DB[/yellow]")
        return
    if dry_run:
        console.print("[dim]dry-run, 未写 DB[/dim]")
        return

    # 拿现 run 内最大 person_idx, 新 person 从 +1 开始
    max_idx = conn.execute(
        "SELECT MAX(person_idx) AS m FROM persons WHERE run_id = ?", (run_id,)
    ).fetchone()["m"] or 0

    new_ids: dict[int, int] = {}  # sub_idx -> new persons.id
    for sub_idx, centroid in result.centroids.items():
        max_idx += 1
        size = int((result.labels == sub_idx).sum())
        new_pid = db.insert_person(conn, run_id, max_idx, size, centroid)
        new_ids[sub_idx] = new_pid
        console.print(f"  + new person_idx={max_idx} ({size} 张脸)")

    # 移转 face_samples
    for face_id, label in zip(face_ids, result.labels, strict=True):
        target_pid = new_ids.get(int(label))
        if target_pid is None:
            continue  # 噪声留在原 person 还是删? 先删, 让 user 决定
            # 也可: conn.execute("DELETE FROM face_samples WHERE face_id=? AND person_id=?", (face_id, pdb))
        conn.execute(
            "UPDATE face_samples SET person_id = ?, is_manual = 1 WHERE face_id = ? AND person_id = ?",
            (target_pid, face_id, pdb),
        )

    # 原 person size 重算 (可能还剩噪声 face)
    new_size = conn.execute(
        "SELECT COUNT(*) AS n FROM face_samples WHERE person_id = ?", (pdb,)
    ).fetchone()["n"]
    conn.execute("UPDATE persons SET size = ? WHERE id = ?", (new_size, pdb))
    console.print(f"[green]✓[/green] 拆分完成. 原 person {person_idx} 剩 {new_size} 张")


@app.command(name="push-jellyfin")
def cmd_push_jellyfin(
    jf_url: Annotated[str, typer.Option("--url", help="Jellyfin 地址", envvar="JF_URL")] = "http://10.100.100.13:8096",
    api_key: Annotated[str, typer.Option("--api-key", help="Jellyfin API key", envvar="JF_API_KEY")] = "",
    db_path: DBOpt = Path("./face_cast.db"),
    run_id: Annotated[int | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="覆盖已有头像")] = False,
    dry_run: bool = False,
):
    """把 named person 的代表头像直接 POST 到 Jellyfin Persons API.

    Jellyfin 在 .actors/ 自动发现常常不灵, 用此命令直接灌. 前提是 NFO 已经写入
    且 Jellyfin 扫过库 (这样 Person entity 已存在).
    """
    if not api_key:
        raise typer.BadParameter("--api-key 必填 (或设 JF_API_KEY 环境变量)")
    conn = db.connect(db_path.resolve())
    if run_id is None:
        active = db.active_run(conn)
        if active is None:
            raise typer.BadParameter("没有 active run")
        run_id = active["id"]
    jf = jellyfin.JellyfinClient(base_url=jf_url, api_key=api_key)
    stats = jellyfin.push_named_persons(conn, run_id, jf, overwrite=overwrite, dry_run=dry_run)
    console.print(
        f"\n[bold]总计[/bold]: {stats['persons']} person · "
        f"上传 {stats['uploaded']} · 跳过 {stats['skipped']} · "
        f"Jellyfin 没此人 {stats['missing_in_jf']} · 无 crop {stats['no_crop']} · 失败 {stats['failed']}"
    )


@app.command(name="ui")
def cmd_ui(
    db_path: DBOpt = Path("./face_cast.db"),
    host: Annotated[str, typer.Option("--host", help="bind 地址")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="端口")] = 9100,
):
    """启动 web UI (浏览器里整理 cluster: 命名/合并/拆分/推 Jellyfin)."""
    from .web.app import serve  # noqa: PLC0415
    serve(db_path.resolve(), host=host, port=port)


@app.command(name="stats")
def cmd_stats(db_path: DBOpt = Path("./face_cast.db")):
    """整体统计: 视频/帧/脸/embedding/person 数."""
    conn = db.connect(db_path.resolve())
    queries = {
        "videos": "SELECT COUNT(DISTINCT video_path) FROM frames",
        "frames": "SELECT COUNT(*) FROM frames",
        "faces": "SELECT COUNT(*) FROM faces",
        "embeddings": "SELECT COUNT(*) FROM embeddings",
        "models": "SELECT COUNT(DISTINCT model_name||model_version) FROM embeddings",
        "detection_runs": "SELECT COUNT(*) FROM detection_runs",
        "active_run": "SELECT id FROM detection_runs WHERE is_active=1 LIMIT 1",
    }
    table = Table(show_header=False)
    for k, q in queries.items():
        v = conn.execute(q).fetchone()
        table.add_row(k, str(v[0] if v else "—"))
    console.print(table)


def cli() -> None:
    """``face-cast`` 入口 (pyproject 注册)."""
    app()


if __name__ == "__main__":
    cli()
