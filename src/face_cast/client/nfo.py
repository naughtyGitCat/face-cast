"""NFO 改写 — 在已有 .nfo (Kodi/Jellyfin movie 格式) 上加/换 <actor> 字段.

策略:
  - 找到 video 同名 .nfo (xxx.mp4 → xxx.nfo)
  - 解析现有 XML, 保留所有非 actor 字段
  - 删掉旧 <actor> 块
  - 按聚类结果插入新 <actor>
  - 写回, 保持原编码 (UTF-8 with declaration)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree


@dataclass
class ActorInfo:
    name: str          # 'cluster_42' 或 '演员A' (用户命名后)
    role: str = ""     # 角色, 通常空
    type: str = "Actor"  # Kodi 约定


def nfo_path_for(video_path: Path) -> Path:
    """xxx.mp4 → xxx.nfo (同目录, 同 stem)."""
    return video_path.with_suffix(".nfo")


def update_actors(
    nfo_path: Path,
    actors: list[ActorInfo],
    *,
    create_if_missing: bool = False,
) -> bool:
    """读取/解析/改写. 成功返回 True. 文件不存在且 create_if_missing=False 返回 False."""
    if not nfo_path.is_file():
        if not create_if_missing:
            return False
        # 创建一个最小骨架
        root = etree.Element("movie")
        etree.SubElement(root, "title").text = nfo_path.stem
    else:
        try:
            # 兼容部分 NFO 没有 XML 声明
            parser = etree.XMLParser(remove_blank_text=False, recover=True)
            tree = etree.parse(str(nfo_path), parser=parser)
            root = tree.getroot()
            if root is None or root.tag != "movie":
                # tag 不对就重建, 老内容可能被破坏
                root = etree.Element("movie")
                etree.SubElement(root, "title").text = nfo_path.stem
        except etree.XMLSyntaxError:
            return False

    # 删除现有 <actor>
    for elem in root.findall("actor"):
        root.remove(elem)

    # 按 actors 列表插入 (顺序 = 主角度排序)
    for a in actors:
        ae = etree.SubElement(root, "actor")
        etree.SubElement(ae, "name").text = a.name
        if a.role:
            etree.SubElement(ae, "role").text = a.role
        etree.SubElement(ae, "type").text = a.type

    # 序列化, UTF-8 + 声明 + pretty
    etree.indent(root, space="  ")
    xml_bytes = etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        standalone=True,
    )
    nfo_path.write_bytes(xml_bytes)
    return True
