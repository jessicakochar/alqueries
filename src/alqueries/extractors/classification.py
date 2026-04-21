# alqueries/extractors/classification.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alqueries.extractors.base import FeatureExtractor


class ClassificationFeatureExtractor(FeatureExtractor):
    """
    Extracts logits, probs, and (optionally) embeddings from a classification
    model over a pool dataloader. Returns everything aligned to loader order.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        embedding_layer: str | None = None,
        input_key: str | int = 0,
    ):
        super().__init__(model, device)
        self._embedding_layer = embedding_layer
        self._input_key = input_key  # how to pull inputs from a batch

    def _inputs_from_batch(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            return batch[self._input_key]
        if isinstance(batch, dict):
            return batch[self._input_key]
        return batch  # assume it's already a tensor

    def extract(self, loader: DataLoader) -> dict[str, np.ndarray | torch.Tensor]:
        self._model.eval()

        # optional embedding hook
        embeddings_chunks: list[torch.Tensor] = []
        hook_handle = None
        if self._embedding_layer is not None:
            target = dict(self._model.named_modules()).get(self._embedding_layer)
            if target is None:
                raise ValueError(
                    f"Layer '{self._embedding_layer}' not found in model."
                )
            hook_handle = target.register_forward_hook(
                lambda _m, _in, out: embeddings_chunks.append(
                    out.detach().cpu().flatten(1)
                )
            )

        logits_chunks: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for batch in loader:
                    inputs = self._inputs_from_batch(batch).to(
                        self._device, non_blocking=True
                    )
                    logits_chunks.append(self._model(inputs).detach().cpu())
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        logits = torch.cat(logits_chunks, dim=0)
        out: dict[str, np.ndarray | torch.Tensor] = {
            "logits": logits,
            "probs": F.softmax(logits, dim=1),
        }
        if embeddings_chunks:
            out["embeddings"] = torch.cat(embeddings_chunks, dim=0).numpy()
        return out

    def extract_mc(
        self, loader: DataLoader, n_runs: int = 10, reduce: str = "mean"
    ) -> torch.Tensor:
        """Monte Carlo dropout probs. reduce='mean' -> (N,C); 'none' -> (R,N,C)."""
        self._model.train()  # keep dropout on
        runs: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(n_runs):
                logits_chunks: list[torch.Tensor] = []
                for batch in loader:
                    inputs = self._inputs_from_batch(batch).to(
                        self._device, non_blocking=True
                    )
                    logits_chunks.append(self._model(inputs).detach().cpu())
                runs.append(F.softmax(torch.cat(logits_chunks, dim=0), dim=1))
        stacked = torch.stack(runs, dim=0)
        return stacked.mean(0) if reduce == "mean" else stacked