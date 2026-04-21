# alqueries/strategies/kmeans.py
from __future__ import annotations

import numpy as np

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("kmeans")
class KMeansSampling(QueryStrategy):
    def __init__(
        self,
        pca_dim: int | None = 128,
        cast_to_float16: bool = True,
        kmeans_kwargs: dict | None = None,
    ):
        self._pca_dim = pca_dim
        self._cast_to_float16 = cast_to_float16
        self._kmeans_kwargs = kmeans_kwargs or {"verbose": 1, "n_init": 10}

    def query(
        self,
        pool_indices: np.ndarray,
        n_samples: int,
        *,
        embeddings: np.ndarray,
        **_,
    ) -> np.ndarray:
        """
        embeddings: (N, D) feature vectors, row i aligned to pool_indices[i].
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        if self._pca_dim is not None and embeddings.shape[1] > self._pca_dim:
            embeddings = PCA(n_components=self._pca_dim).fit_transform(embeddings)
        if self._cast_to_float16:
            embeddings = embeddings.astype(np.float16)

        km = KMeans(n_clusters=n_samples, **self._kmeans_kwargs).fit(embeddings)
        cluster_ids = km.predict(embeddings)
        centers = km.cluster_centers_[cluster_ids]
        dist = ((embeddings - centers) ** 2).sum(axis=1)

        picked = np.array([
            np.flatnonzero(cluster_ids == i)[dist[cluster_ids == i].argmin()]
            for i in range(n_samples)
        ])
        return pool_indices[picked]