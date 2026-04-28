"""HDBSCAN 聚类 — 输入 (face_id, embedding) 矩阵, 输出 cluster 分配."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray         # shape (N,), 每个 face 的 cluster_idx (-1=噪声)
    n_clusters: int
    n_noise: int
    centroids: dict[int, np.ndarray]  # cluster_idx -> 该 cluster 的 embedding 均值


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    metric: str = "cosine",
) -> ClusterResult:
    """对 N×D embedding 矩阵跑 HDBSCAN. 返回每条记录的 cluster id."""
    import hdbscan  # noqa: PLC0415

    if embeddings.shape[0] == 0:
        return ClusterResult(np.zeros(0, dtype=int), 0, 0, {})

    # cosine 距离需要先做 L2 normalize, 然后用 euclidean (HDBSCAN 不直接支持 cosine on cdist 上 condense_tree)
    # 实际等价: cosine_dist(a,b) = 1 - dot(a̅, b̅), 当 a̅,b̅ 单位向量时与 0.5*||a̅-b̅||² 单调
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
    return ClusterResult(
        labels=labels.astype(int),
        n_clusters=len(unique),
        n_noise=n_noise,
        centroids=centroids,
    )
