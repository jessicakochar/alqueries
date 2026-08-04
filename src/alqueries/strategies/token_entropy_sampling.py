from __future__ import annotations

import numpy as np

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("token_entropy_sampling")
class TokenEntropySampling(QueryStrategy):
    def query(
        self,
        *,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        mean_token_uncertainty: np.ndarray,
        **_,
    ) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        if len(unlabeled_indices) == 0:
            return np.array([], dtype=np.int64)

        if n_samples > len(unlabeled_indices):
            raise ValueError(
                "n_samples cannot exceed the number of unlabeled samples."
            )

        scores = np.asarray(mean_token_uncertainty)

        if scores.ndim != 1:
            raise ValueError("mean_token_uncertainty must be one-dimensional.")

        if len(scores) <= int(np.max(unlabeled_indices)):
            raise ValueError(
                "mean_token_uncertainty must contain one score per dataset sample."
            )

        unlabeled_scores = scores[unlabeled_indices]
        relative_selected = np.argsort(unlabeled_scores)[-n_samples:][::-1]
        return unlabeled_indices[relative_selected]
