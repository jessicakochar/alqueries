# alqueries/strategies/entropy.py
from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


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
        probs = probs[unlabeled_indices]
        log_probs = torch.log(probs.clamp_min(1e-12))
        uncertainties = (probs * log_probs).sum(1)
        return unlabeled_indices[uncertainties.sort()[1][:n_samples]]
