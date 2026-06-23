from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy
from alqueries.strategies.utils import max_confidence, mean_dropout_probs, select_by_score


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
        uncertainties = max_confidence(mean_dropout_probs(mc_probs, unlabeled_indices))
        return select_by_score(unlabeled_indices, uncertainties, n_samples)
