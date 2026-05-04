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

        pool_probs = probs[:, unlabeled_indices, :]
        predicted_classes = pool_probs.argmax(dim=2)
        T = predicted_classes.shape[0]
        M = predicted_classes.shape[1]

        f_max = torch.zeros(M, dtype=torch.float32)
        for i in range(M):
            counts = torch.bincount(predicted_classes[:, i], minlength=pool_probs.shape[2])
            f_max[i] = counts.max().float()

        scores = 1.0 - (f_max / T)

        return unlabeled_indices[scores.argsort(descending=True)[:n_samples]]