from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from alqueries import QueryEngine
from alqueries.extractors.classification import ClassificationFeatureExtractor


N_POOL, N_FEATURES, N_CLASSES = 100, 16, 5


def make_tensor_pool() -> TensorDataset:
    return TensorDataset(
        torch.randn(N_POOL, N_FEATURES),
        torch.randint(0, N_CLASSES, (N_POOL,)),
    )


def make_mlp_classifier() -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_FEATURES, 32),
        nn.ReLU(),
        nn.Linear(32, N_CLASSES),
    )


def make_classification_engine(labeled_count: int = 20) -> QueryEngine:
    dataset = make_tensor_pool()
    extractor = ClassificationFeatureExtractor(
        model=make_mlp_classifier(),
        device="cpu",
        embedding_layer="0",
        input_key=0,
    )
    return QueryEngine(dataset, labeled_indices=np.arange(labeled_count), extractor=extractor)
