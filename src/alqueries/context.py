# alqueries/context.py
from __future__ import annotations

from functools import cached_property
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from alqueries.base import ModelFn


class DataloaderScoringContext:
    """
    Default ScoringContext: given a dataset, the absolute indices to score,
    a torch model, and a model_fn(batch, model) -> logits callable, it runs
    inference once and caches logits / probs / embeddings.

    The ordering contract is enforced here: we build a sequential DataLoader
    over Subset(dataset, pool_indices). Row i of every returned tensor
    corresponds to pool_indices[i].
    """

    def __init__(
        self,
        dataset: Dataset,
        pool_indices: np.ndarray,
        model: torch.nn.Module,
        model_fn: Callable[[object, torch.nn.Module], torch.Tensor],
        device: torch.device | str = "cpu",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn=None,
    ):
        self._dataset = dataset
        self._pool_indices = np.asarray(pool_indices, dtype=np.int64)
        self._model = model
        self._model_fn = model_fn
        self._device = torch.device(device) if isinstance(device, str) else device
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._collate_fn = collate_fn

    # ------------------------------------------------------------------ #
    # public ScoringContext surface
    # ------------------------------------------------------------------ #

    @property
    def pool_indices(self) -> np.ndarray:
        return self._pool_indices

    def logits(self) -> torch.Tensor:
        return self._logits

    def probs(self) -> torch.Tensor:
        return F.softmax(self._logits, dim=1)

    def mc_probs(self, n_runs: int = 10, reduce: str = "mean") -> torch.Tensor:
        runs = torch.stack([
            F.softmax(self._forward_pass(allow_dropout=True), dim=1)
            for _ in range(n_runs)
        ])
        return runs.mean(0) if reduce == "mean" else runs

    def embeddings(self, layer: str | None = None) -> torch.Tensor:
        if layer is None:
            raise ValueError("embeddings() requires `layer` (module name).")
        target = dict(self._model.named_modules()).get(layer)
        if target is None:
            raise ValueError(f"Layer '{layer}' not found in model.")

        collected: list[torch.Tensor] = []
        handle = target.register_forward_hook(
            lambda _m, _in, out: collected.append(out.detach().cpu().flatten(1))
        )
        try:
            self._forward_pass(allow_dropout=False)  # triggers the hook
        finally:
            handle.remove()
        return torch.cat(collected, dim=0)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    @cached_property
    def _loader(self) -> DataLoader:
        subset = Subset(self._dataset, self._pool_indices.tolist())
        return DataLoader(
            subset,
            batch_size=self._batch_size,
            shuffle=False,                  # CRITICAL: preserves index alignment
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    @cached_property
    def _logits(self) -> torch.Tensor:
        return self._forward_pass(allow_dropout=False)

    def _forward_pass(self, allow_dropout: bool) -> torch.Tensor:
        if allow_dropout:
            self._model.train()
        else:
            self._model.eval()
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in self._loader:
                logits = self._model_fn(batch, self._model)
                chunks.append(logits.detach().cpu())
        return torch.cat(chunks, dim=0)