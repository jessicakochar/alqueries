from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from alqueries.huggingface.cord import IGNORE_INDEX, predict_layoutlmv3_token_features


class TinyCordDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        labels = torch.tensor([0, IGNORE_INDEX, 1], dtype=torch.long)
        return {
            "input_ids": torch.tensor([101, 102, 0], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 0], dtype=torch.long),
            "bbox": torch.zeros(3, 4, dtype=torch.long),
            "pixel_values": torch.zeros(3, 224, 224),
            "labels": labels,
            "sample_index": index,
        }


class TinyLayoutLMv3Model(torch.nn.Module):
    def forward(self, **kwargs):
        assert "labels" not in kwargs
        batch_size, sequence_length = kwargs["input_ids"].shape
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
                [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]],
            ],
            dtype=torch.float32,
        )[:batch_size, :sequence_length]
        hidden = torch.ones(batch_size, sequence_length, 4)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


def test_layoutlmv3_token_features_skip_labels_and_return_document_uncertainty():
    features = predict_layoutlmv3_token_features(
        TinyLayoutLMv3Model(),
        TinyCordDataset(),
        batch_size=2,
    )

    assert features["probs"].shape == (2, 2)
    assert features["embeddings"].shape == (2, 4)
    assert len(features["token_probs"]) == 2
    assert len(features["token_uncertainties"]) == 2
    assert features["mean_token_uncertainty"].shape == (2,)
    assert features["max_token_uncertainty"].shape == (2,)
