from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy
from alqueries.strategies.utils import resolve_mc_probs, select_by_score


@register_strategy("mean_std")
class MeanStd(QueryStrategy):
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
        sigma_c = torch.std(pool_probs, dim=0, unbiased=False)
        uncertainties = sigma_c.mean(dim=1)
        return select_by_score(unlabeled_indices, uncertainties, n_samples, descending=True)
