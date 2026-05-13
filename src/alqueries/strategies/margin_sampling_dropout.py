from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("margin_sampling_dropout")
class MarginSamplingDropout(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        mc_probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        probs_sorted, _ = mc_probs.mean(dim=0)[unlabeled_indices].sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        idx = uncertainties.argsort()[:n_samples]
        return np.asarray(unlabeled_indices[idx.detach().cpu().numpy()])
