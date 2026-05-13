import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


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
        pool_probs = probs[unlabeled_indices]
        top2_probs, _ = torch.topk(pool_probs, k=2, dim=1)
        margins = top2_probs[:, 0] - top2_probs[:, 1]
        return unlabeled_indices[margins.argsort()[:n_samples]]





    # # Step 1: Sort probabilities in descending order
    # sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

    # # Step 2: Compute margin (top1 - top2)
    # margins = sorted_probs[:, 0] - sorted_probs[:, 1]

    # margins = top2_probs[:, 0] - top2_probs[:, 1]

    # # Step 3: Get indices of smallest margins (most uncertain)
    # query_indices = torch.argsort(margins)

    # # Step 4: Select top n_query samples
    # return query_indices[:n_query]
