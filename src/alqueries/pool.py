from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import numpy as np
from alqueries.base import QueryStrategy
from alqueries.extractors.base import FeatureExtractor
from alqueries.registry import get_strategy

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class QueryEngine:
    """
    Orchestrates pool bookkeeping and strategy execution.

    It owns one full dataset and explicit labeled/unlabeled index splits,
    then resolves the requested subset before calling a strategy.

    If an extractor is configured, features are always extracted once over the
    full dataset and passed through to the strategy call as kwargs.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        labeled_indices: np.ndarray | list[int] | None = None,
        extractor: FeatureExtractor | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        dataloader_kwargs: dict | None = None,
    ):
        self._dataset = dataset
        self._extractor = extractor
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloader_kwargs = dataloader_kwargs or {}

        full = np.arange(len(dataset), dtype=np.int64) # type: ignore[assignment]
        labeled = np.array([], dtype=np.int64)
        if labeled_indices is not None:
            labeled = np.unique(np.asarray(labeled_indices, dtype=np.int64))
        unlabeled = np.setdiff1d(full, labeled, assume_unique=True)

        self._full_indices = full
        self._labeled_indices = labeled
        self._unlabeled_indices = unlabeled

        self._labeled_mask = np.zeros(len(self._dataset), dtype=bool) # type: ignore[assignment]
        self._labeled_mask[self._labeled_indices] = True

        # just make sure labelled and unlabelled indices are consistent with the full set and mask
        assert set(self._labeled_indices.tolist()).issubset(set(self._full_indices.tolist()))
        assert set(self._unlabeled_indices.tolist()).issubset(set(self._full_indices.tolist()))
        assert set(self._labeled_indices.tolist()).union(set(self._unlabeled_indices.tolist())) == set(self._full_indices.tolist())
        assert set(self._labeled_indices.tolist()).isdisjoint(set(self._unlabeled_indices.tolist()))
        assert np.all(self._labeled_mask[self._labeled_indices])
        assert not np.any(self._labeled_mask[self._unlabeled_indices])

    def query(
        self,
        strategy: QueryStrategy,
        n_samples: int,
    ) -> np.ndarray:
        from torch.utils.data import DataLoader, Subset

        auto_features: dict = {}
        if self._extractor is not None:
            full_loader = DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                **self._dataloader_kwargs,
            )
            auto_features = self._extractor.extract(full_loader)

        return strategy.query(
            labeled_indices=self._labeled_indices,
            unlabeled_indices=self._unlabeled_indices,
            labeled_mask=self._labeled_mask,
            n_samples=n_samples,
            **auto_features,
        )
