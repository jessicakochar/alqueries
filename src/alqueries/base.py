# alqueries/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch


# A model_fn is any callable that maps a batch -> logits tensor.
# The caller decides what "batch" means — tuple, tensor, dict — and adapts
# model_fn accordingly. Strategies never see the batch.
ModelFn = Callable[[object], torch.Tensor]


@runtime_checkable
class ScoringContext(Protocol):
    """
    Everything a query strategy needs. All returned tensors are aligned to
    `pool_indices`: row i corresponds to pool_indices[i] in the full train set.
    """

    @property
    def pool_indices(self) -> np.ndarray: ...

    def logits(self) -> torch.Tensor: ...
    def probs(self) -> torch.Tensor: ...
    def mc_probs(self, n_runs: int = 10, reduce: str = "mean") -> torch.Tensor: ...
    def embeddings(self, layer: str | None = None) -> torch.Tensor: ...


class QueryStrategy(ABC):
    @abstractmethod
    def query(self, ctx: ScoringContext, n_samples: int) -> np.ndarray:
        """Return `n_samples` absolute indices from the unlabeled pool."""