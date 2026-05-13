import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


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
        probs = probs[unlabeled_indices]
        preds = torch.max(probs, dim=1)[0]
        uncertainties = 1.0 - preds
        return unlabeled_indices[uncertainties.sort(descending=True)[1][:n_samples]]
