import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from alqueries import QueryEngine
from alqueries.base import QueryStrategy
from alqueries.extractors.base import FeatureExtractor


class RecordingExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(model=nn.Identity(), device="cpu")
        self.dataset_sizes: list[int] = []

    def extract(self, loader):
        self.dataset_sizes.append(len(loader.dataset))
        n = len(loader.dataset)
        probs = torch.full((n, 2), 0.5, dtype=torch.float32)
        return {"probs": probs}


class EchoUnlabeledStrategy(QueryStrategy):
    def query(self, n_samples, *, unlabeled_indices, labeled_mask, probs, **_):
        assert probs.shape[0] == labeled_mask.shape[0]
        assert labeled_mask.dtype == np.bool_
        return unlabeled_indices[:n_samples]


def test_query_engine_passes_unlabeled_split_and_features():
    dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    extractor = RecordingExtractor()
    engine = QueryEngine(dataset, labeled_indices=[0, 2], extractor=extractor, batch_size=3)

    picked = engine.query(EchoUnlabeledStrategy(), n_samples=3)

    assert picked.tolist() == [1, 3, 4]
    assert extractor.dataset_sizes == [8]


def test_query_engine_without_extractor_still_passes_split_metadata():
    dataset = TensorDataset(torch.randn(6, 2), torch.randint(0, 2, (6,)))
    engine = QueryEngine(dataset, labeled_indices=[0, 1])

    class UsesOnlySplit(QueryStrategy):
        def query(self, n_samples, *, unlabeled_indices, labeled_indices, **_):
            assert set(labeled_indices.tolist()) == {0, 1}
            return unlabeled_indices[:n_samples]

    picked = engine.query(UsesOnlySplit(), n_samples=2)
    assert picked.tolist() == [2, 3]
