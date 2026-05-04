
import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("bayesian_active_learning_disagreement_dropout")
class BayesianALDropout(QueryStrategy):

    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        epsilon = 1e-10

        pool_probs = probs[:, unlabeled_indices, :]        # (T, M, C)

        mean_probs = pool_probs.mean(dim=0)                # (M, C)
        H_mean = -(mean_probs * (mean_probs + epsilon).log()).sum(dim=1)  # (M,)

        H_each = -(pool_probs * (pool_probs + epsilon).log()).sum(dim=2)  # (T, M)
        mean_H = H_each.mean(dim=0)                        # (M,)

        bald_scores = H_mean - mean_H                      # (M,)

        # Step 5 — highest BALD score = most informative to query.
        return unlabeled_indices[bald_scores.argsort(descending=True)[:n_samples]]