"""face-cast 客户端配置.

只放本地秘密 / 常用默认值 (e.g. Jellyfin API key, base URL).
不要塞进 SQLite, 也不要进 git. 默认查找顺序:

  1. ``--config`` 显式指定的路径
  2. ``$FACE_CAST_CONFIG`` 环境变量
  3. ``<db_dir>/config.toml``  ← 推荐, 跟 DB 一起部署
  4. ``~/.config/face-cast/config.toml``

文件格式 (TOML):

    [jellyfin]
    url = "http://10.100.100.13:8096"
    api_key = "xxxxxxxxxxxxxxxx"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JellyfinConfig:
    url: str = ""
    api_key: str = ""

    @property
    def configured(self) -> bool:
        """有 api_key 才算配好 (URL 有兜底默认值)."""
        return bool(self.api_key)


@dataclass
class ServerConfig:
    """face-cast inference server (跑在 GPU 机, 默认 .11:8001)."""
    url: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.url)


@dataclass
class ScanConfig:
    """detect 默认参数."""
    frames: int = 15


@dataclass
class Config:
    jellyfin: JellyfinConfig = field(default_factory=JellyfinConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    source_path: Path | None = None  # 给调试 / UI 显示用

    @classmethod
    def load(cls, path: Path | None = None, db_path: Path | None = None) -> "Config":
        """按顺序找配置文件; 找不到就返回空 Config (不报错)."""
        candidates: list[Path] = []
        if path is not None:
            candidates.append(Path(path))
        env = os.environ.get("FACE_CAST_CONFIG")
        if env:
            candidates.append(Path(env))
        if db_path is not None:
            candidates.append(Path(db_path).parent / "config.toml")
        candidates.append(Path.home() / ".config" / "face-cast" / "config.toml")

        for cand in candidates:
            if cand.is_file():
                return cls._from_file(cand)
        return cls()

    @classmethod
    def _from_file(cls, p: Path) -> "Config":
        with p.open("rb") as f:
            data = tomllib.load(f)
        jf = data.get("jellyfin", {}) or {}
        sv = data.get("server", {}) or {}
        sc = data.get("scan", {}) or {}
        return cls(
            jellyfin=JellyfinConfig(
                url=str(jf.get("url", "") or ""),
                api_key=str(jf.get("api_key", "") or ""),
            ),
            server=ServerConfig(
                url=str(sv.get("url", "") or ""),
            ),
            scan=ScanConfig(
                frames=int(sc.get("frames", 15) or 15),
            ),
            source_path=p,
        )
