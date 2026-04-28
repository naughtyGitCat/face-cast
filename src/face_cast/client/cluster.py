"""HDBSCAN 聚类算法封装 — 把 N×D embedding 矩阵分到「同一个 person」.

文件名保留 cluster.py (反映底层算法是 HDBSCAN 聚类),
但语义层面输出的是 "person 列表".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """一次 person 识别的结果."""

    labels: np.ndarray         # shape (N,), 每个 face 的 person_idx (-1=噪声)
    n_persons: int
    n_noise: int
    centroids: dict[int, np.ndarray]  # person_idx -> 该 person 的 embedding 均值


def detect_persons(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    metric: str = "cosine",
) -> DetectionResult:
    """对 N×D embedding 矩阵跑 HDBSCAN. 返回每条记录的 person_idx."""
    import hdbscan  # noqa: PLC0415

    if embeddings.shape[0] == 0:
        return DetectionResult(np.zeros(0, dtype=int), 0, 0, {})

    # cosine 距离: 先 L2 normalize, 再用 euclidean (HDBSCAN 不直接支持 cosine)
    # cosine_dist(a,b) = 1 - dot(â, b̂); 单位向量下与 ||â-b̂||² 单调
    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = (embeddings / norms).astype(np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # 已 L2 normalize, 等价 cosine
            cluster_selection_method="eom",
        )
    else:
        X = embeddings.astype(np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method="eom",
        )

    labels = clusterer.fit_predict(X)
    unique = sorted({int(c) for c in labels if c >= 0})
    centroids = {}
    for c in unique:
        centroids[c] = embeddings[labels == c].mean(axis=0).astype(np.float32)
    n_noise = int((labels == -1).sum())
    return DetectionResult(
        labels=labels.astype(int),
        n_persons=len(unique),
        n_noise=n_noise,
        centroids=centroids,
    )


def split_subcluster(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 2,
) -> DetectionResult:
    """对一个 person 的 embedding 子集再聚类, 找出"其实是多人"的情况.

    跟 detect_persons 一致, 但参数更宽松 (默认 min_cluster=2).
    """
    return detect_persons(
        embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
    )
