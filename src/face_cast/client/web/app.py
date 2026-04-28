"""face-cast UI — 浏览/合并/拆分/命名 person, 推 Jellyfin 头像.

设计:
  - bottle + jinja2, 零 npm 构建
  - HTMX 做局部刷新, 不写前端 JS 框架
  - 直接读写 SQLite (跟 face-cast 客户端共用 DB)
  - 不依赖 face-cast inference server (.11 GPU 关机也能用)

启动:
  face-cast ui --db E:\\face-cast\\face_cast.db --port 9100
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote as urlquote

from bottle import Bottle, HTTPResponse, redirect, request, response, static_file
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import db, fs
from ..cluster import split_subcluster
from ..config import Config
from ..jellyfin import JellyfinClient, push_named_persons
from ..scan_jobs import JobManager


_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ─── bottle UTF-8 helper ─────────────────────────────────────────────
# bottle 0.13 对 application/x-www-form-urlencoded body 和 QUERY_STRING 默认
# 按 latin1 decode (BaseRequest.POST / BaseRequest.query). 直接用 .get() 拿到
# 的中文是双字节假 latin1 str, 写 SQLite 会双重编码乱码.
# 必须走 .getunicode() 让 bottle 内部做 s.encode('latin1').decode('utf-8')
# 把 hack 还原. 全部 user input 都通过这两个 helper 读, 防止漏掉.
def _q(name: str, default: str = "") -> str:
    """读 ?foo=... query string, UTF-8 安全."""
    return request.query.getunicode(name, encoding="utf-8") or default


def _f(name: str, default: str = "") -> str:
    """读 POST form body, UTF-8 安全."""
    return request.forms.getunicode(name, encoding="utf-8") or default


def _find_face_cast_bin() -> Path:
    """找 face-cast 可执行文件, 用来 spawn detect/cluster 子进程.

    优先 .venv/Scripts/face-cast.exe (NSSM 服务安装的位置), fallback 到
    sys.executable 同目录, 再 fallback 到 python -m face_cast.client.cli.
    """
    here = Path(sys.executable).parent
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [here / f"face-cast{suffix}"]
    for c in candidates:
        if c.exists():
            return c
    # fallback: 用 sys.executable -m
    return Path(sys.executable)


def create_app(db_path: Path, config: Config | None = None) -> Bottle:
    app = Bottle()
    db_path = Path(db_path)
    cfg = config or Config()

    # job runner: 把 jobs 元数据 + log 放在 db 同目录的 jobs/
    job_mgr = JobManager(db_path.parent / "jobs")
    face_cast_bin = _find_face_cast_bin()

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        enable_async=False,
    )
    env.globals["app_title"] = "face-cast UI"
    env.globals["cfg"] = cfg
    # urlquote: 模板里 path 进 query string 必须先 encode (中文/反斜杠)
    env.filters["urlquote"] = lambda s: urlquote(str(s), safe="")

    def render(name: str, **ctx: Any) -> str:
        return env.get_template(name).render(**ctx)

    def conn() -> sqlite3.Connection:
        return db.connect(db_path)

    def active_run_id(c: sqlite3.Connection, override: int | None = None) -> int:
        if override is not None:
            return override
        a = db.active_run(c)
        if a is None:
            raise HTTPResponse(body="<h1>没有 active run</h1><p>先跑 face-cast detect</p>", status=404)
        return a["id"]

    # ─── 主页: person 列表 ─────────────────────────────────────────────

    @app.get("/")
    def index():
        c = conn()
        run_id = active_run_id(c)
        run = c.execute("SELECT * FROM cluster_runs_view WHERE id=?",
                        (run_id,)).fetchone() if False else c.execute(
            "SELECT * FROM detection_runs WHERE id=?", (run_id,)
        ).fetchone()
        persons = db.list_persons(c, run_id, include_noise=False)
        # 给每个 person 拿一张代表 face_id (排序用 det_score 最高)
        for p in persons:
            row = c.execute(
                """
                SELECT fa.id FROM faces fa
                JOIN face_samples fs ON fs.face_id = fa.id
                WHERE fs.person_id = ? AND fa.crop_jpeg IS NOT NULL
                ORDER BY fa.det_score DESC LIMIT 1
                """,
                (p["id"],),
            ).fetchone()
            p["thumb_face_id"] = row["id"] if row else None
        # 统计
        stats = {
            "videos": c.execute("SELECT COUNT(DISTINCT video_path) AS n FROM frames").fetchone()["n"],
            "faces": c.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"],
            "persons_named": c.execute(
                "SELECT COUNT(*) AS n FROM persons WHERE run_id=? AND display_name IS NOT NULL",
                (run_id,),
            ).fetchone()["n"],
            "persons_total": len(persons),
        }
        return render("index.html", run=dict(run), persons=persons, stats=stats)

    # ─── Person 详情 ──────────────────────────────────────────────────

    @app.get("/person/<idx:int>")
    def person_detail(idx: int):
        c = conn()
        run_id = active_run_id(c)
        pdb = db.person_db_id(c, run_id, idx)
        if pdb is None:
            raise HTTPResponse(body=f"<h1>person {idx} not found</h1>", status=404)
        person = c.execute(
            "SELECT * FROM persons WHERE id = ?", (pdb,)
        ).fetchone()
        person = dict(person)
        faces = db.person_face_ids(c, pdb, limit=200)
        videos = db.person_video_summary(c, pdb)
        similar = db.top_similar_persons(c, run_id, idx, k=8, min_similarity=0.2)
        # 给 similar 也补 thumb
        for s in similar:
            s_pdb = db.person_db_id(c, run_id, s["person_idx"])
            row = c.execute(
                "SELECT fa.id FROM faces fa JOIN face_samples fs ON fs.face_id=fa.id "
                "WHERE fs.person_id=? AND fa.crop_jpeg IS NOT NULL "
                "ORDER BY fa.det_score DESC LIMIT 1",
                (s_pdb,),
            ).fetchone()
            s["thumb_face_id"] = row["id"] if row else None
        return render("person.html", person=person, faces=faces, videos=videos,
                      similar=similar, run_id=run_id)

    # ─── face crop 图片 ───────────────────────────────────────────────

    @app.get("/face/<face_id:int>.jpg")
    def face_image(face_id: int):
        c = conn()
        crop = db.face_crop(c, face_id)
        if crop is None:
            raise HTTPResponse(status=404)
        response.set_header("Content-Type", "image/jpeg")
        response.set_header("Cache-Control", "public, max-age=3600")
        return crop

    # ─── 操作: 命名 ───────────────────────────────────────────────────

    @app.post("/person/<idx:int>/name")
    def name_person(idx: int):
        c = conn()
        run_id = active_run_id(c)
        new_name = _f("display_name").strip()
        c.execute(
            "UPDATE persons SET display_name = ? WHERE run_id = ? AND person_idx = ?",
            (new_name or None, run_id, idx),
        )
        # HTMX 用 HX-Refresh 刷新整个页面
        response.set_header("HX-Refresh", "true")
        return ""

    # ─── 操作: 合并 ───────────────────────────────────────────────────

    @app.post("/person/<idx:int>/merge")
    def merge_into(idx: int):
        """body: source_idx 必填, 把它合并到当前 idx (target)."""
        c = conn()
        run_id = active_run_id(c)
        src_str = _f("source_idx")
        try:
            sources = [int(x) for x in src_str.split(",") if x.strip()]
        except ValueError:
            raise HTTPResponse(status=400, body="source_idx 必须是逗号分隔整数")
        if not sources:
            raise HTTPResponse(status=400, body="source_idx 不能为空")
        result = db.merge_persons(c, run_id=run_id, target_idx=idx,
                                   source_indices=sources, delete_sources=True)
        response.set_header("HX-Refresh", "true")
        return f"merged {result['moved_face_samples']} faces"

    # ─── 操作: 拆分 ───────────────────────────────────────────────────

    @app.post("/person/<idx:int>/split")
    def split_person(idx: int):
        import numpy as np  # noqa: PLC0415
        c = conn()
        run_id = active_run_id(c)
        pdb = db.person_db_id(c, run_id, idx)
        if pdb is None:
            raise HTTPResponse(status=404)
        rows = list(c.execute(
            "SELECT fs.face_id, e.vector, e.dim FROM face_samples fs "
            "JOIN embeddings e ON e.face_id = fs.face_id "
            "JOIN persons p ON p.id = fs.person_id "
            "WHERE fs.person_id=? AND e.model_name=(SELECT model_name FROM detection_runs WHERE id=p.run_id)",
            (pdb,),
        ))
        if not rows:
            return "no embeddings"
        dim = rows[0]["dim"]
        face_ids = [r["face_id"] for r in rows]
        matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(-1, dim)
        result = split_subcluster(matrix, min_cluster_size=2, min_samples=2)
        if result.n_persons == 0:
            return "0 sub-clusters; nothing changed"

        max_idx = c.execute(
            "SELECT MAX(person_idx) AS m FROM persons WHERE run_id=?", (run_id,),
        ).fetchone()["m"] or 0
        new_ids: dict[int, int] = {}
        for sub_idx, centroid in result.centroids.items():
            max_idx += 1
            size = int((result.labels == sub_idx).sum())
            new_pid = db.insert_person(c, run_id, max_idx, size, centroid)
            new_ids[sub_idx] = new_pid
        for face_id, label in zip(face_ids, result.labels, strict=True):
            target_pid = new_ids.get(int(label))
            if target_pid is None:
                continue
            c.execute(
                "UPDATE face_samples SET person_id=?, is_manual=1 WHERE face_id=? AND person_id=?",
                (target_pid, face_id, pdb),
            )
        new_size = c.execute(
            "SELECT COUNT(*) AS n FROM face_samples WHERE person_id=?", (pdb,),
        ).fetchone()["n"]
        c.execute("UPDATE persons SET size=? WHERE id=?", (new_size, pdb))
        response.set_header("HX-Refresh", "true")
        return f"split into {result.n_persons} new persons"

    # ─── 操作: 踢脸 ───────────────────────────────────────────────────

    @app.post("/face/<face_id:int>/eject")
    def eject_face_(face_id: int):
        c = conn()
        run_id = active_run_id(c)
        # 找出这 face 当前所属的 person (在 active run)
        row = c.execute(
            "SELECT fs.person_id FROM face_samples fs JOIN persons p ON p.id=fs.person_id "
            "WHERE fs.face_id=? AND p.run_id=? LIMIT 1",
            (face_id, run_id),
        ).fetchone()
        if row is None:
            raise HTTPResponse(status=404)
        db.eject_face(c, face_id, row["person_id"])
        response.set_header("HX-Refresh", "true")
        return "ejected"

    # ─── 操作: 推 Jellyfin ────────────────────────────────────────────

    @app.post("/person/<idx:int>/push-jellyfin")
    def push_one(idx: int):
        c = conn()
        run_id = active_run_id(c)
        # form value > config > 兜底默认; 配了 config.toml 就不用每次填表单
        url = _f("jf_url") or cfg.jellyfin.url or "http://10.100.100.13:8096"
        api_key = _f("jf_api_key") or cfg.jellyfin.api_key
        if not api_key:
            return "missing api_key (填表单或在 config.toml [jellyfin].api_key)"
        # 限定到这一个 person
        person = c.execute(
            "SELECT * FROM persons WHERE run_id=? AND person_idx=?",
            (run_id, idx),
        ).fetchone()
        if not person or not person["display_name"]:
            return "person 没起名, 先 /person/<idx>/name"
        # 复用 push_named_persons 但只跑这一个
        # (jellyfin 模块按 SELECT * FROM persons 跑全部, 这里 hack: 用 sub-query 限定)
        # 简单做法: 临时把别的 named person 拿出来, 跑完恢复. 太复杂. 直接调底层.
        from ..jellyfin import JellyfinClient as _JC  # noqa: PLC0415
        rep = db.representative_face(c, person["id"])
        if rep is None:
            return "no crop"
        crop, _ = rep
        jf = _JC(base_url=url, api_key=api_key)
        jf_id = jf.find_person(person["display_name"])
        if jf_id is None:
            return f"Jellyfin 里没找到 person '{person['display_name']}'"
        ok, msg = jf.upload_primary(jf_id, crop)
        return f"uploaded {len(crop)} B ({msg})" if ok else f"upload failed: {msg}"

    @app.post("/api/push-all")
    def push_all():
        c = conn()
        run_id = active_run_id(c)
        url = _f("jf_url") or cfg.jellyfin.url or "http://10.100.100.13:8096"
        api_key = _f("jf_api_key") or cfg.jellyfin.api_key
        overwrite = bool(request.forms.get("overwrite"))
        if not api_key:
            return "missing api_key (填表单或在 config.toml [jellyfin].api_key)"
        jf = JellyfinClient(base_url=url, api_key=api_key)
        stats = push_named_persons(c, run_id, jf, overwrite=overwrite)
        return f"{stats}"

    # ─── Folder tree ──────────────────────────────────────────────────

    @app.get("/folders")
    def folders_index():
        c = conn()
        run_id = active_run_id(c)
        items = db.folder_summary(c, run_id)
        return render("folders.html", folders=items, run_id=run_id)

    @app.get("/folder")
    def folder_detail():
        c = conn()
        run_id = active_run_id(c)
        path = _q("path")
        if not path:
            redirect("/folders")
        videos = db.folder_videos(c, run_id, path)
        return render("folder.html", folder=path, videos=videos, run_id=run_id)

    # ─── 批量合并候选 ─────────────────────────────────────────────────

    @app.get("/candidates")
    def candidates():
        c = conn()
        run_id = active_run_id(c)
        try:
            min_sim = float(_q("min", "0.45"))
        except ValueError:
            min_sim = 0.45
        pairs = db.candidate_pairs(c, run_id, min_similarity=min_sim, limit=200)
        # 给每个 person 找 thumb
        thumb_cache: dict[int, int | None] = {}
        for p in pairs:
            for side in ("a", "b"):
                pid = p[f"{side}_id"]
                if pid not in thumb_cache:
                    row = c.execute(
                        "SELECT id FROM faces fa JOIN face_samples fs ON fs.face_id=fa.id "
                        "WHERE fs.person_id=? AND fa.crop_jpeg IS NOT NULL "
                        "ORDER BY fa.det_score DESC LIMIT 1",
                        (pid,),
                    ).fetchone()
                    thumb_cache[pid] = row["id"] if row else None
                p[f"{side}_thumb"] = thumb_cache[pid]
        return render("candidates.html", pairs=pairs, min_sim=min_sim, run_id=run_id)

    # ─── 本机磁盘浏览 ─────────────────────────────────────────────────
    # 纯 HTML5 风格: 每点一次都跳一次新页, breadcrumb 导航, 不依赖 JS.

    def _breadcrumbs(path: str) -> list[tuple[str, str]]:
        """把 'F:\\China\\王东瑶' 拆成 [('F:\\', 'F:'), ('F:\\China', 'China'), ...]"""
        if not path:
            return []
        sep = "\\" if "\\" in path else "/"
        parts = [p for p in path.rstrip(sep).split(sep) if p]
        crumbs: list[tuple[str, str]] = []
        if os.name == "nt" and parts and parts[0].endswith(":"):
            # Windows: 第一段是 'F:', 盘根需要 'F:\\' 才能 os.scandir
            crumbs.append((parts[0] + sep, parts[0]))
            for i in range(1, len(parts)):
                full = parts[0] + sep + sep.join(parts[1:i + 1])
                crumbs.append((full, parts[i]))
        else:
            for i, name in enumerate(parts):
                crumbs.append((sep + sep.join(parts[:i + 1]), name))
        return crumbs

    @app.get("/browse")
    def browse():
        path = _q("path")
        if path:
            entries = fs.list_dirs(path)
        else:
            entries = fs.list_drives()
        # 给每个 entry 补已扫描视频数 (递归)
        c = conn()
        rich_entries = []
        for e in entries:
            scanned = db.folder_scan_count(c, e.path)
            rich_entries.append({
                "name": e.name,
                "path": e.path,
                "has_subdirs": e.has_subdirs,
                "video_count": e.video_count,
                "scanned": scanned,
            })
        # 当前 path 的 scanned 总数 (作为页面 header 信息)
        path_scanned = db.folder_scan_count(c, path) if path else 0
        return render(
            "browse.html",
            path=path,
            entries=rich_entries,
            breadcrumbs=_breadcrumbs(path),
            path_scanned=path_scanned,
            server_configured=cfg.server.configured,
        )

    @app.post("/api/scan")
    def start_scan():
        path = _f("path")
        if not path:
            raise HTTPResponse(status=400, body="missing path")
        if not Path(path).is_dir():
            raise HTTPResponse(status=400, body=f"not a directory: {path}")
        if not cfg.server.configured:
            raise HTTPResponse(
                status=400,
                body="config.toml [server].url 没配, 不知道 inference server 地址",
            )
        info = job_mgr.start_extract(
            face_cast_bin=face_cast_bin,
            path=path,
            db_path=db_path,
            server_url=cfg.server.url,
            frames=cfg.scan.frames,
        )
        redirect(f"/jobs#{info.id}")

    @app.post("/api/cluster")
    def start_cluster():
        if not cfg.server.configured:
            raise HTTPResponse(
                status=400,
                body="config.toml [server].url 没配",
            )
        info = job_mgr.start_cluster(
            face_cast_bin=face_cast_bin,
            db_path=db_path,
            server_url=cfg.server.url,
        )
        if info is None:
            raise HTTPResponse(status=409, body="已有 cluster job 在跑, 等它完")
        redirect(f"/jobs#{info.id}")

    @app.get("/jobs/<job_id>")
    def job_detail(job_id: str):
        j = job_mgr.get(job_id)
        if j is None:
            raise HTTPResponse(status=404, body=f"job {job_id} not found")
        full_log = job_mgr.read_log_full(job_id)
        view = {
            "id": j.id,
            "kind": j.kind,
            "target": j.target,
            "status": j.status,
            "elapsed": j.elapsed,
            "returncode": j.returncode,
            "pid": j.pid,
            "started_at": j.started_at,
            "ended_at": j.ended_at,
            "cmd": " ".join(j.cmd),
            "log": full_log,
            "log_path": j.log_path,
        }
        return render("job_detail.html", job=view, is_running=(j.status == "running"))

    @app.get("/jobs/<job_id>/log")
    def job_log_raw(job_id: str):
        """完整日志 text/plain. ``?tail=N`` 只要末 N 行."""
        j = job_mgr.get(job_id)
        if j is None:
            raise HTTPResponse(status=404)
        try:
            n = int(_q("tail", "0"))
        except ValueError:
            n = 0
        if n > 0:
            text = job_mgr.tail_log(job_id, lines=n)
        else:
            text = job_mgr.read_log_full(job_id)
        response.content_type = "text/plain; charset=utf-8"
        return text

    @app.get("/jobs")
    def jobs_index():
        jobs = job_mgr.list_jobs()
        # dataclass → dict + tail, 给模板用
        rich = [
            {
                "id": j.id,
                "kind": j.kind,
                "target": j.target,
                "status": j.status,
                "elapsed": j.elapsed,
                "returncode": j.returncode,
                "tail": job_mgr.tail_log(j.id, lines=30),
                "cmd": " ".join(j.cmd),
            }
            for j in jobs
        ]
        return render("jobs.html", jobs=rich, has_running=job_mgr.has_running())

    # ─── 静态资源 ─────────────────────────────────────────────────────

    @app.get("/static/<filename:path>")
    def static(filename: str):
        return static_file(filename, root=str(_STATIC_DIR))

    return app


def serve(db_path: Path, host: str = "0.0.0.0", port: int = 9100,
          server: str = "waitress", config_path: Path | None = None) -> None:
    cfg = Config.load(path=config_path, db_path=db_path)
    app = create_app(db_path, cfg)
    cfg_info = (f" config={cfg.source_path}"
                if cfg.source_path is not None else " config=<none>")
    jf_info = (f" jellyfin={cfg.jellyfin.url} (key set)"
               if cfg.jellyfin.configured else " jellyfin=<not configured>")
    print(f"[face-cast UI] db={db_path}{cfg_info}{jf_info}", flush=True)
    print(f"[face-cast UI] listening on http://{host}:{port}", flush=True)
    if server == "waitress":
        try:
            from waitress import serve as waitress_serve  # noqa: PLC0415
            waitress_serve(app, host=host, port=port, threads=4)
            return
        except ImportError:
            pass
    import bottle  # noqa: PLC0415
    bottle.run(app, host=host, port=port, server="wsgiref", quiet=True)
