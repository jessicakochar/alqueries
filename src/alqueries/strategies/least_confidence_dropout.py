from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("least_confidence_dropout")
class LeastConfidenceDropoutSampling(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        mc_probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        probs = mc_probs.mean(dim=0)[unlabeled_indices]
        uncertainties = probs.max(dim=1)[0]
        return unlabeled_indices[uncertainties.sort()[1][:n_samples]]
