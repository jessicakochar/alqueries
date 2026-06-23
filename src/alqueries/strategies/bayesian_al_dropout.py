from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy
from alqueries.strategies.utils import resolve_mc_probs, select_by_score


@register_strategy("bald_dropout")
@register_strategy("bayesian_active_learning_disagreement_dropout")
class BayesianALDropout(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        mc_probs: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        **_,
    ) -> np.ndarray:
        mc_probs = resolve_mc_probs(mc_probs, probs)
        pool_probs = mc_probs[:, unlabeled_indices, :]
        probs_mean = pool_probs.mean(dim=0)
        predictive_entropy = (-probs_mean * torch.log(probs_mean.clamp_min(1e-12))).sum(dim=1)
        expected_entropy = (-pool_probs * torch.log(pool_probs.clamp_min(1e-12))).sum(dim=2).mean(dim=0)
        uncertainties = expected_entropy - predictive_entropy
        return select_by_score(unlabeled_indices, uncertainties, n_samples)
