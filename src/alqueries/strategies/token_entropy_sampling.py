from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("token_entropy_sampling")
class TokenEntropySampling(QueryStrategy):
    def query(
        self,
        *,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        token_probs: torch.Tensor,
        valid_token_mask: torch.Tensor,
        **_,
    ) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        if len(unlabeled_indices) == 0:
            return np.array([], dtype=np.int64)

        if n_samples > len(unlabeled_indices):
            raise ValueError(
                "n_samples cannot exceed the number of unlabeled samples."
            )

        if token_probs.ndim != 3:
            raise ValueError("token_probs must have shape (N, S, C).")

        if valid_token_mask.ndim != 2:
            raise ValueError("valid_token_mask must have shape (N, S).")

        if token_probs.shape[:2] != valid_token_mask.shape:
            raise ValueError("token_probs and valid_token_mask must align.")

        if token_probs.shape[0] <= int(np.max(unlabeled_indices)):
            raise ValueError(
                "token_probs must contain one row per dataset sample."
            )

        entropy = -(
            token_probs
            * torch.log(token_probs.clamp_min(1e-12))
        ).sum(dim=-1)
        mask = valid_token_mask.to(dtype=entropy.dtype)
        scores = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        unlabeled_scores = scores[unlabeled_indices]
        if torch.is_tensor(unlabeled_scores):
            unlabeled_scores = unlabeled_scores.detach().cpu().numpy()

        relative_selected = np.argsort(unlabeled_scores)[-n_samples:][::-1]
        return unlabeled_indices[relative_selected]
