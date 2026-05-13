from __future__ import annotations
import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy

@register_strategy("entropy_sampling_dropout")
class EntropySamplingDropout(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        mc_probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        probs = mc_probs.mean(dim=0)[unlabeled_indices]
        log_probs = torch.log(probs.clamp_min(1e-12))
        uncertainties = (probs * log_probs).sum(dim=1)
        idx = uncertainties.sort()[1][:n_samples]
        return np.asarray(unlabeled_indices[idx.detach().cpu().numpy()])
