from __future__ import annotations
import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy

@register_strategy("least_confidence")
class LeastConfidenceSampling(QueryStrategy):

    def query(self, unlabeled_indices: np.ndarray, n_samples: int, *, probs: torch.Tensor, **_,) -> np.ndarray:
        # Slice to only the unlabelled pool rows — same pattern as entropy.
        probs = probs[unlabeled_indices]

        max_probs, _ = probs.max(dim=1)
        uncertainties = 1.0 - max_probs
        # Most uncertain = highest score. argsort descending, take first n.
        return unlabeled_indices[uncertainties.argsort(descending=True)[:n_samples]]


