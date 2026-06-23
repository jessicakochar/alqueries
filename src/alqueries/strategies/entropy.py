# alqueries/strategies/entropy.py
from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy
from alqueries.strategies.utils import negative_entropy, pool_rows, select_by_score


@register_strategy("entropy_sampling")
@register_strategy("entropy")
class EntropySampling(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        """
        probs: (N, C) softmax probabilities, row i aligned to unlabeled_indices[i].
        """
        uncertainties = negative_entropy(pool_rows(probs, unlabeled_indices))
        return select_by_score(unlabeled_indices, uncertainties, n_samples)
