import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alqueries.extractors.classification import (
    ClassificationFeatureExtractor,
    HuggingFaceClassificationFeatureExtractor,
)


def test_extract_returns_logits_probs_and_embeddings():
    n_pool, n_features, n_classes = 12, 8, 4
    dataset = TensorDataset(
        torch.randn(n_pool, n_features),
        torch.randint(0, n_classes, (n_pool,)),
    )
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    model = nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )
    extractor = ClassificationFeatureExtractor(
        model=model,
        device="cpu",
        embedding_layer="0",
        input_key=0,
    )

    out = extractor.extract(loader)

    assert set(out.keys()) == {"logits", "probs", "embeddings", "mc_logits", "mc_probs"}
    assert out["logits"].shape == (n_pool, n_classes)
    assert out["probs"].shape == (n_pool, n_classes)
    assert out["embeddings"].shape == (n_pool, 16)
    np.testing.assert_allclose(out["probs"].sum(dim=1).numpy(), 1.0, rtol=1e-6, atol=1e-6)


def test_extract_raises_when_embedding_layer_is_missing():
    dataset = TensorDataset(torch.randn(4, 3), torch.randint(0, 2, (4,)))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(3, 2))

    extractor = ClassificationFeatureExtractor(
        model=model,
        device="cpu",
        embedding_layer="does_not_exist",
        input_key=0,
    )

    with pytest.raises(ValueError, match="not found"):
        extractor.extract(loader)


class TinyHFClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = input_ids.float()
        logits = self.linear(embeddings)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "embeddings": embeddings}


def test_huggingface_classification_extractor_returns_general_features():
    loader = DataLoader(
        [
            {
                "input_ids": torch.tensor([1.0, 0.0]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor(0),
            },
            {
                "input_ids": torch.tensor([0.0, 1.0]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor(1),
            },
        ],
        batch_size=2,
    )
    extractor = HuggingFaceClassificationFeatureExtractor(
        model=TinyHFClassifier(),
        device="cpu",
        mc_dropout_runs=2,
    )

    out = extractor.extract(loader)

    assert set(out.keys()) == {"logits", "probs", "embeddings", "mc_logits", "mc_probs"}
    assert out["logits"].shape == (2, 2)
    assert out["probs"].shape == (2, 2)
    assert out["embeddings"].shape == (2, 2)
    assert out["mc_logits"].shape == (2, 2, 2)
    assert out["mc_probs"].shape == (2, 2, 2)
