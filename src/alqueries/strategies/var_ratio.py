import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy
from alqueries.strategies.utils import max_confidence, pool_rows, select_by_score


@register_strategy("var_ratio")
class VarRatio(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        uncertainties = 1.0 - max_confidence(pool_rows(probs, unlabeled_indices))
        return select_by_score(unlabeled_indices, uncertainties, n_samples, descending=True)
