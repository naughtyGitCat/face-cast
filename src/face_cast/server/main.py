"""face-cast 服务端 — 无状态人脸推理服务 (bottle 版).

设计:
  - 不存任何图像/embedding/历史
  - 每个请求: image bytes -> faces[] (bbox, embedding, kps, age, sex)
  - 启动时载入 InsightFace buffalo_l 到 GPU, 后续纯推理
  - 客户端负责所有持久化与聚类

启动:
  python -m face_cast.server.main --host 0.0.0.0 --port 9000

Endpoints:
  GET  /health        探活
  GET  /model/info    返回当前模型 + 版本 + dim, 客户端用此打 embedding 标签
  POST /detect        单图 (multipart file=...)
  POST /detect_batch  批量 (multipart files=[...])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _register_nvidia_dlls() -> None:
    """让 onnxruntime-gpu 找到 nvidia-cudnn-cu12 / nvidia-cublas-cu12 等 pip 包里
    带的 DLL — Windows 不会自动搜 site-packages 的 nvidia/*/bin/.

    必须在 import insightface / onnxruntime 之前调用. (insightface 是 lifespan
    内延迟导入的, 但放这里更保险)
    """
    if sys.platform != "win32":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return
    for sub in nvidia_root.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, FileNotFoundError):
                pass


_register_nvidia_dlls()

import bottle
import cv2
import numpy as np
from bottle import Bottle, HTTPError, request, response

# ─── 模型元信息 (随版本一起改) ─────────────────────────────────────────────
MODEL_NAME = os.environ.get("FACE_MODEL_NAME", "buffalo_l")
MODEL_VERSION = os.environ.get("FACE_MODEL_VERSION", "2024.01.15-onnx")
DETECTOR = os.environ.get("FACE_DETECTOR", "retinaface_buffalo_l")
EMBEDDING_DIM = 512  # buffalo_l 固定; 换模型时记得同步
DET_SIZE = (640, 640)


app = Bottle()
_state: dict = {}


# ─── idle suspend (省电用) ───────────────────────────────────────────────
# 配置走 env var, 不写 config 文件因为 server 端尽量保持无状态.
#   FACE_IDLE_SUSPEND_SECS   超过这么久没请求就 suspend (默认 0 = 关闭)
#   FACE_IDLE_SUSPEND_MODE   "hibernate" (S4) 或 "sleep" (S3); 默认
#                            hibernate (零功耗, ~30s 唤醒); 注意 S4 需要先
#                            ``powercfg /hibernate on`` 启用
#
# 实现:
#   - bottle ``before_request`` hook 在每个请求处理前 bump _last_request
#   - daemon thread 每 30s 检查 idle, 超时调 SetSuspendState
#   - SetSuspendState 阻塞到系统真的睡下去并被唤醒, 返回时把 _last_request
#     重置避免立刻又睡

import threading  # noqa: E402

_last_request_lock = threading.Lock()
_last_request_at = time.time()


def _bump_activity() -> None:
    global _last_request_at
    with _last_request_lock:
        _last_request_at = time.time()


@app.hook("before_request")
def _hook_bump():  # noqa: ANN202
    _bump_activity()


def _suspend(mode: str) -> bool:
    """调 SetSuspendState 进 sleep (S3) 或 hibernate (S4). 阻塞到唤醒.

    LocalSystem 默认有 SeShutdownPrivilege, 不用手动 AdjustTokenPrivileges.
    """
    if sys.platform != "win32":
        print("[idle-suspend] non-Windows, 跳过", file=sys.stderr, flush=True)
        return False
    import ctypes  # noqa: PLC0415
    hibernate = 1 if mode == "hibernate" else 0
    force = 1
    disable_wake_event = 0
    try:
        ok = ctypes.windll.powrprof.SetSuspendState(
            hibernate, force, disable_wake_event)
        return bool(ok)
    except OSError as e:
        print(f"[idle-suspend] SetSuspendState err: {e}", file=sys.stderr, flush=True)
        return False


def _idle_watcher(timeout_s: int, mode: str) -> None:
    print(f"[idle-suspend] watcher 启动: 阈值 {timeout_s}s, 模式 {mode}",
          flush=True)
    poll_interval = max(15, min(60, timeout_s // 4))
    while True:
        time.sleep(poll_interval)
        with _last_request_lock:
            idle = time.time() - _last_request_at
        if idle < timeout_s:
            continue
        print(f"[idle-suspend] {idle:.0f}s 无请求 → {mode}", flush=True)
        _suspend(mode)
        # SetSuspendState 阻塞返回时系统已经唤醒. 重置计时, 否则下一轮
        # 立刻又触发.
        _bump_activity()
        print("[idle-suspend] 系统已唤醒, 计时重置", flush=True)


def _start_idle_watcher_if_configured() -> None:
    raw = os.environ.get("FACE_IDLE_SUSPEND_SECS", "").strip()
    if not raw:
        return
    try:
        timeout = int(raw)
    except ValueError:
        print(f"[idle-suspend] FACE_IDLE_SUSPEND_SECS={raw!r} 不是整数, 跳过",
              file=sys.stderr, flush=True)
        return
    if timeout <= 0:
        return
    mode = os.environ.get("FACE_IDLE_SUSPEND_MODE", "hibernate").lower()
    if mode not in {"hibernate", "sleep"}:
        print(f"[idle-suspend] FACE_IDLE_SUSPEND_MODE={mode!r} 非法, 用 hibernate",
              file=sys.stderr, flush=True)
        mode = "hibernate"
    t = threading.Thread(target=_idle_watcher, args=(timeout, mode), daemon=True)
    t.start()


# ─── helpers ─────────────────────────────────────────────────────────────


def _decode(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPError(400, "decode failed (not a valid image)")
    return img


def _faces_to_dict(faces) -> list[dict]:
    """InsightFace Face objects → JSON-friendly dicts."""
    out = []
    for f in faces:
        # 性别归一化: InsightFace 0.7.3 返回 'F'/'M' 字符 (.sex), 不是 int
        # 也有些版本提供 .gender (int 0=female, 1=male). 都尝试.
        sex_val: int | None = None
        if hasattr(f, "gender") and f.gender is not None:
            sex_val = int(f.gender)
        elif hasattr(f, "sex") and f.sex is not None:
            sex_val = {"F": 0, "f": 0, "M": 1, "m": 1}.get(str(f.sex), None)
        out.append(
            {
                "bbox": f.bbox.astype(int).tolist(),       # [x1, y1, x2, y2]
                "kps": f.kps.astype(float).tolist(),       # 5 个关键点 [(x,y)*5]
                "embedding": f.embedding.astype(float).tolist(),  # 512 float
                "det_score": float(f.det_score),
                "age": int(f.age) if f.age is not None else None,
                "sex": sex_val,
            }
        )
    return out


def _ok(payload: dict) -> str:
    response.content_type = "application/json; charset=utf-8"
    return json.dumps(payload, ensure_ascii=False)


# ─── routes ──────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return _ok({
        "ok": "fa" in _state,
        "model_name": MODEL_NAME,
        "providers": _state.get("providers"),
    })


@app.get("/model/info")
def model_info():
    """客户端启动时调一次, 用返回值标 embedding 的 model_name / model_version."""
    return _ok({
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "detector": DETECTOR,
        "embedding_dim": EMBEDDING_DIM,
        "det_size": list(DET_SIZE),
    })


@app.post("/detect")
def detect():
    """单图: 检测 + embedding."""
    t0 = time.time()
    upload = request.files.get("file")
    if upload is None:
        raise HTTPError(400, "missing 'file' multipart field")
    raw = upload.file.read()
    img = _decode(raw)
    faces = _state["fa"].get(img)
    return _ok({
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "image_shape": [img.shape[1], img.shape[0]],   # [w, h]
        "faces": _faces_to_dict(faces),
        "took_ms": int((time.time() - t0) * 1000),
    })


@app.post("/detect_batch")
def detect_batch():
    """批量: 一次发 N 张, 减少 HTTP 往返. 推荐 8-16 张/批.

    multipart 字段名都用 ``files``, bottle 用 getall 拿全部.
    """
    t0 = time.time()
    uploads = request.files.getall("files")
    if not uploads:
        raise HTTPError(400, "missing 'files' multipart field (at least one)")
    results = []
    fa = _state["fa"]
    for f in uploads:
        try:
            raw = f.file.read()
            img = _decode(raw)
            faces = fa.get(img)
            results.append({
                "filename": f.raw_filename or f.filename,
                "image_shape": [img.shape[1], img.shape[0]],
                "faces": _faces_to_dict(faces),
            })
        except HTTPError as e:
            results.append({"filename": f.raw_filename or f.filename, "error": str(e.body)})
        except Exception as e:  # noqa: BLE001
            results.append({"filename": f.raw_filename or f.filename, "error": str(e)})
    return _ok({
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "results": results,
        "took_ms": int((time.time() - t0) * 1000),
    })


# ─── lifecycle ───────────────────────────────────────────────────────────


def _load_model() -> None:
    """同步载入 (启动时调一次, 主线程阻塞至 GPU 暖好).

    模型根目录优先级:
      1. ``FACE_MODEL_ROOT`` 环境变量 (服务化场景, 让 LocalSystem 找到模型)
      2. ``~/.insightface`` (默认, 单用户)
    """
    from insightface.app import FaceAnalysis  # noqa: PLC0415

    model_root = os.environ.get("FACE_MODEL_ROOT")
    where = model_root or "~/.insightface"
    print(f"[face-cast] loading {MODEL_NAME} (root={where}) ...", flush=True)
    t0 = time.time()
    kwargs: dict = {
        "name": MODEL_NAME,
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    if model_root:
        kwargs["root"] = model_root
    fa = FaceAnalysis(**kwargs)
    fa.prepare(ctx_id=0, det_size=DET_SIZE)
    _state["fa"] = fa
    _state["providers"] = fa.models["detection"].session.get_providers()
    print(
        f"[face-cast] ready in {time.time() - t0:.1f}s, providers={_state['providers']}",
        flush=True,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────


def cli() -> None:
    """``face-server`` 入口 (pyproject 注册)."""
    p = argparse.ArgumentParser(prog="face-server", description="face-cast 推理服务")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument(
        "--server",
        default="waitress",
        choices=["waitress", "wsgiref", "auto"],
        help="WSGI server (waitress 推荐, wsgiref 仅调试用)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=2,
        help="waitress 线程数. GPU 推理本质串行, 设小值即可",
    )
    args = p.parse_args()

    _load_model()  # 启动前先把模型暖好, 第一次请求不卡
    _start_idle_watcher_if_configured()

    if args.server == "waitress":
        try:
            from waitress import serve  # noqa: PLC0415
        except ImportError:
            print("waitress 未安装, 落到 wsgiref. 装: pip install waitress", file=sys.stderr)
            args.server = "wsgiref"

    if args.server == "waitress":
        from waitress import serve  # noqa: PLC0415
        print(f"[face-cast] waitress on http://{args.host}:{args.port}  (threads={args.threads})", flush=True)
        serve(app, host=args.host, port=args.port, threads=args.threads)
    else:
        print(f"[face-cast] wsgiref on http://{args.host}:{args.port}", flush=True)
        bottle.run(app, host=args.host, port=args.port, server="wsgiref", quiet=True)


if __name__ == "__main__":
    cli()
