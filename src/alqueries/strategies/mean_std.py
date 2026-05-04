import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("mean_std")
class MeanStd(QueryStrategy):

    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        pool_probs = probs[:, unlabeled_indices, :]       # (T, M, C)
        std_probs = torch.std(pool_probs, dim=0)          # (M, C)
        scores = std_probs.sum(dim=1)
                        # (M,)
        return unlabeled_indices[scores.argsort(descending=True)[:n_samples]]