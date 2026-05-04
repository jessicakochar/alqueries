import torch
import numpy as np
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("margin_sampling_dropout")
class MarginSamplingDropout(QueryStrategy):
    def query(self, unlabeled_indices: np.ndarray, n_samples: int, *, probs: torch.Tensor, **_) -> np.ndarray:

        pool_probs = probs[:, unlabeled_indices, :]

        top2 = torch.topk(pool_probs, k=2, dim=2).values
        margins = top2[:, :, 0] - top2[:, :, 1]

        mean_margins = margins.mean(dim=0)

        # Step 4 — smallest average margin = most uncertain.
        return unlabeled_indices[mean_margins.argsort()[:n_samples]]

# def margin_sampling_dropout(probs: torch.Tensor, n_query: int) -> torch.Tensor:

#     # Step 1: Average probabilities across dropout passes
#     # avg_probs = probs.mean(dim=0)  # Shape: (N, C)

#     # Step 2: Compute margin (top1 - top2)
#     top2_probs = torch.topk(probs, k=2, dim=2).values
#     margins = top2_probs[:, :, 0] - top2_probs[:, :, 1]
#     mean_margins = torch.mean(margins, dim=0)  # Average margin across dropout passes

#     # Step 3: Get indices of smallest margins (most uncertain)
#     query_indices = torch.argsort(mean_margins)

#     # Step 4: Select top n_query samples
#     return query_indices[:n_query]