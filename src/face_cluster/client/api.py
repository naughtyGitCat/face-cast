"""HTTP 客户端 — 跟 face-cluster server 说话."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests


@dataclass
class ServerInfo:
    model_name: str
    model_version: str
    detector: str
    embedding_dim: int


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    det_score: float
    age: int | None
    sex: int | None


class FaceClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def model_info(self) -> ServerInfo:
        r = self.session.get(f"{self.base_url}/model/info", timeout=10)
        r.raise_for_status()
        d = r.json()
        return ServerInfo(
            model_name=d["model_name"],
            model_version=d["model_version"],
            detector=d["detector"],
            embedding_dim=d["embedding_dim"],
        )

    def detect(self, jpg: bytes, filename: str = "frame.jpg") -> list[FaceDetection]:
        r = self.session.post(
            f"{self.base_url}/detect",
            files={"file": (filename, jpg, "image/jpeg")},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return [_to_face(f) for f in r.json()["faces"]]

    def detect_batch(
        self, jpgs: list[tuple[str, bytes]]
    ) -> list[list[FaceDetection]]:
        """jpgs = [(filename, jpg_bytes), ...] → 与 jpgs 同长的 list, 每元素是 face 列表."""
        files = [("files", (name, data, "image/jpeg")) for name, data in jpgs]
        r = self.session.post(
            f"{self.base_url}/detect_batch",
            files=files,
            timeout=self.timeout,
        )
        r.raise_for_status()
        out: list[list[FaceDetection]] = []
        for item in r.json()["results"]:
            if item.get("error"):
                out.append([])
            else:
                out.append([_to_face(f) for f in item["faces"]])
        return out


def _to_face(d: dict) -> FaceDetection:
    return FaceDetection(
        bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
        embedding=np.asarray(d["embedding"], dtype=np.float32),
        det_score=d["det_score"],
        age=d.get("age"),
        sex=d.get("sex"),
    )
