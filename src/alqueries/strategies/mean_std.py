from __future__ import annotations

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
        mc_probs: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        **_,
    ) -> np.ndarray:
        if mc_probs is None:
            mc_probs = probs
        pool_probs = mc_probs[:, unlabeled_indices, :]
        sigma_c = torch.std(pool_probs, dim=0, unbiased=False)
        uncertainties = sigma_c.mean(dim=1)
        idx = uncertainties.argsort(descending=True)[:n_samples]
        return np.asarray(unlabeled_indices[idx.detach().cpu().numpy()])
