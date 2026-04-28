"""文件系统遍历 helper, 给 web UI 的目录树用.

只暴露 ``list_drives`` (列盘符 / Unix root) 和 ``list_dirs(path)`` (列子目录).
故意做成同步 + 浅层 — 一次只走一层, 让 HTMX 按需展开.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from pathlib import Path

# 跳过这些名字 (Windows + 各种 SCM/cache)
_SKIP_NAMES = frozenset([
    "$RECYCLE.BIN",
    "System Volume Information",
    "Recovery",
    ".git",
    ".svn",
    ".hg",
    ".cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
])

# 视频后缀, 用来给目录显示 "X 视频" 提示
VIDEO_EXTS = frozenset([
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts",
    ".m4v", ".webm", ".mpg", ".mpeg", ".m2ts", ".rmvb", ".rm",
])


@dataclass
class FsEntry:
    name: str               # 显示名 (e.g. 'F:' or 'China')
    path: str               # 绝对路径, 保留原 OS 分隔符
    has_subdirs: bool       # 是否还能再展开 (UI 决定要不要画 ▶)
    video_count: int        # **直接子节点** 中的视频数 (非递归)


def list_drives() -> list[FsEntry]:
    """Windows 列出存在的盘符 (C:/D:/...). Unix 只返回 ``/``."""
    if os.name == "nt":
        out: list[FsEntry] = []
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if os.path.exists(root):
                out.append(FsEntry(
                    name=f"{letter}:",
                    path=root,
                    has_subdirs=True,   # 不进去 scandir, 假定有
                    video_count=0,      # 盘根直接 scandir 太慢, 不算
                ))
        return out
    return [FsEntry(name="/", path="/", has_subdirs=True, video_count=0)]


def _scan_one_dir(p: Path, *, scan_cap: int = 5000) -> tuple[bool, int]:
    """走一层 ``p``, 返回 (有子目录, 直接视频数). PermissionError 静默吞."""
    has_dirs = False
    videos = 0
    try:
        with os.scandir(p) as it:
            for i, entry in enumerate(it):
                if i >= scan_cap:    # 极大目录就放弃, 防止 UI 卡死
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in _SKIP_NAMES:
                            has_dirs = True
                    else:
                        suf = os.path.splitext(entry.name)[1].lower()
                        if suf in VIDEO_EXTS:
                            videos += 1
                except OSError:
                    continue
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return has_dirs, videos


def list_dirs(path: str) -> list[FsEntry]:
    """列 ``path`` 下的子目录 (按名字字典序). ``path`` 不存在就返回空."""
    p = Path(path)
    if not p.is_dir():
        return []

    out: list[FsEntry] = []
    try:
        with os.scandir(p) as it:
            entries = [e for e in it]
    except (PermissionError, FileNotFoundError, OSError):
        return []

    entries.sort(key=lambda e: e.name.lower())

    for e in entries:
        if e.name in _SKIP_NAMES:
            continue
        if e.name.startswith(".") and e.name != ".":
            continue  # 隐藏目录跳过
        try:
            if not e.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        has_dirs, videos = _scan_one_dir(Path(e.path))
        out.append(FsEntry(
            name=e.name,
            path=e.path,
            has_subdirs=has_dirs,
            video_count=videos,
        ))
    return out
