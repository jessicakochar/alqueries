import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy
from alqueries.strategies.utils import pool_rows, select_by_score, top_two_margin


@register_strategy("margin_sampling")
class MarginSampling(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        margins = top_two_margin(pool_rows(probs, unlabeled_indices))
        return select_by_score(unlabeled_indices, margins, n_samples)
