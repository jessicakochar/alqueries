from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from alqueries.base import QueryStrategy
from alqueries.extractors.base import FeatureExtractor

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class QueryEngine:


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

        self._full_indices = np.arange(len(dataset), dtype=np.int64) # type: ignore[assignment]
        self._set_labeled_indices(labeled_indices)

    @property
    def labeled_indices(self) -> np.ndarray:
        return self._labeled_indices.copy()

    @property
    def unlabeled_indices(self) -> np.ndarray:
        return self._unlabeled_indices.copy()

    @property
    def labeled_mask(self) -> np.ndarray:
        return self._labeled_mask.copy()

    # def add_labeled_indices(self, indices: Sequence[int] | np.ndarray) -> None:
    #     labeled = np.concatenate([
    #         self._labeled_indices,
    #         np.asarray(indices, dtype=np.int64),
    #     ])
    #     self._set_labeled_indices(labeled)

    def add_labeled_indices(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> None:
        new_indices = np.asarray(indices, dtype=np.int64)

        if new_indices.ndim != 1:
            raise ValueError("indices must be one-dimensional.")

        if np.any((new_indices < 0) | (new_indices >= len(self._dataset))):
            raise ValueError("indices must be valid dataset indices.")

        if len(np.unique(new_indices)) != len(new_indices):
            raise ValueError("indices must not contain duplicates.")

        if not np.all(np.isin(new_indices, self._unlabeled_indices)):
            raise ValueError("Only currently unlabeled indices can be added.")

        labeled = np.concatenate([
            self._labeled_indices,
            new_indices,
        ])
        self._set_labeled_indices(labeled)

    def _set_labeled_indices(self, labeled_indices: Sequence[int] | np.ndarray | None) -> None:
        labeled = np.array([], dtype=np.int64)
        if labeled_indices is not None:
            labeled = np.unique(np.asarray(labeled_indices, dtype=np.int64))

        if np.any((labeled < 0) | (labeled >= len(self._dataset))):
            raise ValueError("labeled_indices must be valid dataset indices.")

        self._labeled_indices = labeled
        self._unlabeled_indices = np.setdiff1d(self._full_indices, labeled, assume_unique=True)
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
        *,
        features: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        from torch.utils.data import DataLoader

        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        if n_samples > len(self._unlabeled_indices):
            raise ValueError(
                "n_samples cannot exceed the number of unlabeled samples."
            )

        auto_features: dict[str, Any] = {}

        if self._extractor is not None:
            full_loader = DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                **self._dataloader_kwargs,
            )
            auto_features = self._extractor.extract(full_loader)

        query_features = {
            **auto_features,
            **dict(features or {}),
        }

        selected_indices = np.asarray(
            strategy.query(
                labeled_indices=self._labeled_indices,
                unlabeled_indices=self._unlabeled_indices,
                labeled_mask=self._labeled_mask,
                n_samples=n_samples,
                **query_features,
            ),
            dtype=np.int64,
        )

        if selected_indices.ndim != 1:
            raise ValueError(
                "Strategy must return a one-dimensional array."
            )

        if len(selected_indices) > n_samples:
            raise ValueError("Strategy returned more indices than requested.")

        if len(np.unique(selected_indices)) != len(selected_indices):
            raise ValueError(
                "Strategy returned duplicate indices."
            )
        if not np.all(
            np.isin(selected_indices, self._unlabeled_indices)
        ):
            raise ValueError(
                "Strategy must return indices from the unlabeled pool."
            )

        return selected_indices