# alqueries/strategies/random.py
from __future__ import annotations

import numpy as np

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("random")
class RandomSampling(QueryStrategy):
    def __init__(self, seed: int | None = None):
        self._seed = seed

    def query(self, unlabeled_indices: np.ndarray, n_samples: int, **_) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        n = min(n_samples, len(unlabeled_indices))
        return rng.choice(unlabeled_indices, size=n, replace=False)