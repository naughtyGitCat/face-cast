"""Jellyfin Persons API 推送 — 把 face-cast 命名好的 person 头像直接 POST 到 Jellyfin.

Jellyfin 在某些版本对 ``.actors/<name>.jpg`` 自动发现不可靠 (10.11+ 有时不挂头像).
这个模块直接调 ``POST /Items/{personId}/Images/Primary`` 把 crop_jpeg 灌进去, 立刻生效.

依赖:
  - face-cast DB 里有 named persons (display_name 非空)
  - 这些 person 名字在 Jellyfin 里已经被 NFO 触发为 Person entity
    (即先 face-cast write-nfo + Jellyfin 扫一次)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from urllib.parse import quote

import requests
from rich.console import Console

from . import db

console = Console()


@dataclass
class JellyfinClient:
    base_url: str
    api_key: str
    timeout: int = 15

    def _h(self) -> dict[str, str]:
        return {"X-Emby-Token": self.api_key, "Accept": "application/json"}

    def find_person(self, name: str) -> str | None:
        """按名字精确找 Person, 返回 itemId. 没找到返回 None."""
        r = requests.get(
            f"{self.base_url}/Persons",
            params={"searchTerm": name, "limit": 20},
            headers=self._h(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        for it in r.json().get("Items", []):
            if it.get("Name") == name:
                return it["Id"]
        return None

    def upload_primary(self, person_id: str, jpg_bytes: bytes) -> bool:
        """POST jpg 作为 Primary 图. 返回 True 成功."""
        r = requests.post(
            f"{self.base_url}/Items/{person_id}/Images/Primary",
            data=jpg_bytes,
            headers={**self._h(), "Content-Type": "image/jpeg"},
            timeout=self.timeout * 2,
        )
        return r.ok


def push_named_persons(
    conn: sqlite3.Connection,
    run_id: int,
    jf: JellyfinClient,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """把当前 run 里所有 named person 的代表头像 POST 到 Jellyfin Persons API.

    overwrite=False 时, 如果 Jellyfin 该 person 已经有 Primary 图就跳过.
    """
    persons = conn.execute(
        """
        SELECT id, person_idx, display_name
        FROM persons
        WHERE run_id = ? AND person_idx >= 0 AND display_name IS NOT NULL
        ORDER BY size DESC
        """,
        (run_id,),
    ).fetchall()

    stats = {"persons": len(persons), "uploaded": 0, "skipped": 0, "missing_in_jf": 0, "no_crop": 0, "failed": 0}
    if not persons:
        console.print("[yellow]没有 named person, 先 face-cast name-person 起名[/yellow]")
        return stats

    for p in persons:
        rep = db.representative_face(conn, p["id"])
        if rep is None:
            stats["no_crop"] += 1
            console.print(f"[yellow]{p['display_name']}: 无 crop, 跳过[/yellow]")
            continue
        crop_jpg, _ = rep

        try:
            jf_id = jf.find_person(p["display_name"])
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            console.print(f"[red]{p['display_name']}: find 失败 {e}[/red]")
            continue
        if jf_id is None:
            stats["missing_in_jf"] += 1
            console.print(
                f"[yellow]{p['display_name']}: Jellyfin 里没找到 (先 write-nfo + 扫库)[/yellow]"
            )
            continue

        # 检查是否已有图
        try:
            existing = requests.get(
                f"{jf.base_url}/Items/{jf_id}/Images/Primary",
                headers=jf._h(),
                timeout=5,
                allow_redirects=False,
            )
            has_image = existing.status_code == 200
        except Exception:
            has_image = False
        if has_image and not overwrite:
            stats["skipped"] += 1
            console.print(f"[dim]{p['display_name']}: 已有图, 跳过 (--overwrite 强制)[/dim]")
            continue

        if dry_run:
            console.print(f"[cyan]{p['display_name']}: would upload {len(crop_jpg)} B[/cyan]")
            stats["uploaded"] += 1
            continue

        ok = jf.upload_primary(jf_id, crop_jpg)
        if ok:
            stats["uploaded"] += 1
            console.print(f"[green]✓[/green] {p['display_name']}: {len(crop_jpg)} B")
        else:
            stats["failed"] += 1
            console.print(f"[red]FAIL[/red] {p['display_name']}: upload 失败")

    return stats
