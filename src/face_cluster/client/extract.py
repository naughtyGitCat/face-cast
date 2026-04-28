"""ffmpeg / ffprobe 包装 — 抽帧 + 时长.

策略:
  - ffprobe 拿视频时长
  - 跳过开头/结尾各 10%, 中间均匀采 N 帧
  - ffmpeg seek + 单帧提取, 输出到内存 (jpg bytes)
  - 失败重试一次
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMeta:
    duration_s: float
    width: int
    height: int
    codec: str


def ffprobe(path: Path) -> VideoMeta | None:
    """成功返回 VideoMeta, 失败返回 None (不抛)."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height:format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if r.returncode != 0:
            return None
        d = json.loads(r.stdout)
        s = (d.get("streams") or [{}])[0]
        f = d.get("format") or {}
        if not s or "duration" not in f:
            return None
        return VideoMeta(
            duration_s=float(f["duration"]),
            width=int(s.get("width") or 0),
            height=int(s.get("height") or 0),
            codec=s.get("codec_name") or "",
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError):
        return None


def sample_timestamps(duration_s: float, n: int, edge_margin: float = 0.1) -> list[int]:
    """返回 n 个均匀分布的时间戳 (毫秒), 跳过首尾 edge_margin 比例.

    n=5, duration=600, margin=0.1 → ms timestamps 在 [60s, 540s] 上均匀 5 个点
    """
    if duration_s <= 0:
        return []
    if n <= 0:
        return []
    # 极短视频 (< 30s) 只抽 1 帧, 取中间
    if duration_s < 30:
        return [int(duration_s * 500)]
    a = duration_s * edge_margin
    b = duration_s * (1 - edge_margin)
    if n == 1:
        return [int((a + b) * 500)]  # ms
    step = (b - a) / (n - 1)
    return [int((a + step * i) * 1000) for i in range(n)]


def extract_frame(path: Path, ms: int, max_height: int = 720) -> bytes | None:
    """从视频抽指定毫秒位置的一帧, 返回 jpg bytes (压缩传输). 失败返回 None.

    max_height 用于在 ffmpeg 端 scale 下采样 — 减少传输 + 服务端 decode 成本.
    人脸检测 640px 足够, 720p 留余量.
    """
    seek_s = ms / 1000.0
    vf = f"scale=-2:'min({max_height}\\,ih)'"
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{seek_s:.3f}",
                "-i", str(path),
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "3",
                "-f", "image2pipe", "-vcodec", "mjpeg",
                "pipe:1",
            ],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0 or not r.stdout:
            return None
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def crop_face(
    frame_jpg: bytes, bbox: tuple[int, int, int, int], pad_ratio: float = 0.2,
    out_size: int = 224, jpeg_q: int = 85,
) -> bytes | None:
    """从一张 frame jpg 抠出 bbox 区域 (带 padding), resize 到 out_size, JPEG 编码.

    用于 faces.crop_jpeg 缓存. ~10-30 KB/face.
    """
    arr = np.frombuffer(frame_jpg, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # 等比缩放, 短边到 out_size, 保留更多上下文
    side = min(crop.shape[:2])
    scale = out_size / side
    nw, nh = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return buf.tobytes() if ok else None
