from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from alqueries.extractors import TokenClassificationFeatureExtractor
from alqueries.huggingface.cord import IGNORE_INDEX, evaluate_layoutlmv3_token_classifier


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
    extractor = TokenClassificationFeatureExtractor(TinyLayoutLMv3Model())
    features = extractor.extract(DataLoader(TinyCordDataset(), batch_size=2))

    assert features["probs"].shape == (2, 2)
    assert features["embeddings"].shape == (2, 4)
    assert features["token_logits"].shape == (2, 3, 2)
    assert features["token_probs"].shape == (2, 3, 2)
    assert features["token_embeddings"].shape == (2, 3, 4)
    assert features["valid_token_mask"].shape == (2, 3)


def test_entropy_features_do_not_require_hidden_states():
    class LogitsOnlyModel(TinyLayoutLMv3Model):
        def forward(self, **kwargs):
            assert kwargs["output_hidden_states"] is False
            return SimpleNamespace(logits=super().forward(**kwargs).logits)

    loader = DataLoader(TinyCordDataset(), batch_size=2)
    expected = TokenClassificationFeatureExtractor(TinyLayoutLMv3Model()).extract(loader)
    actual = TokenClassificationFeatureExtractor(LogitsOnlyModel()).extract(loader, include_embeddings=False)
    assert "embeddings" not in actual
    assert "token_embeddings" not in actual
    assert torch.equal(actual["token_probs"], expected["token_probs"])
    assert torch.equal(actual["valid_token_mask"], expected["valid_token_mask"])


def test_layoutlmv3_token_evaluation_ignores_padding_labels():
    metrics = evaluate_layoutlmv3_token_classifier(
        TinyLayoutLMv3Model(),
        TinyCordDataset(),
        label_names=["O", "B-TOTAL"],
        batch_size=2,
    )

    assert metrics["eval_accuracy"] == pytest.approx(0.25)
    assert metrics["eval_micro_f1"] == 0.0
    assert metrics["eval_steps"] == 1.0


class SequenceDataset(Dataset):
    def __init__(self, labels, predictions):
        self.labels = labels
        self.predictions = predictions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.predictions[index]),
            "labels": torch.tensor(self.labels[index]),
        }


class SequenceModel(torch.nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids, num_classes=3).float())


def test_evaluation_scores_complete_entities_not_individual_tokens():
    dataset = SequenceDataset([[1, 2, 0, 1, -100]], [[1, 1, 0, 1, 2]])
    metrics = evaluate_layoutlmv3_token_classifier(
        SequenceModel(), dataset, label_names=["O", "B-MENU.NM", "I-MENU.NM"]
    )
    assert metrics["eval_accuracy"] == pytest.approx(0.75)
    assert metrics["eval_precision"] == pytest.approx(1 / 3)
    assert metrics["eval_recall"] == pytest.approx(1 / 2)
    assert metrics["eval_micro_f1"] == pytest.approx(0.4)


def test_evaluation_preserves_receipt_boundaries_and_seqeval_default_mode():
    dataset = SequenceDataset([[1, 2], [2, 2]], [[2, 2], [0, 0]])
    metrics = evaluate_layoutlmv3_token_classifier(
        SequenceModel(), dataset, label_names=["O", "B-MENU.NM", "I-MENU.NM"], batch_size=2
    )
    assert metrics["eval_precision"] == 1.0
    assert metrics["eval_recall"] == 0.5
    assert metrics["eval_micro_f1"] == pytest.approx(2 / 3)


def test_evaluation_rejects_empty_references_and_non_bio_labels():
    with pytest.raises(ValueError, match="no valid labeled tokens"):
        evaluate_layoutlmv3_token_classifier(
            SequenceModel(), SequenceDataset([[-100]], [[0]]),
            label_names=["O", "B-MENU.NM", "I-MENU.NM"],
        )
    with pytest.raises(ValueError, match="BIO"):
        evaluate_layoutlmv3_token_classifier(
            SequenceModel(), TinyCordDataset(), label_names=["menu.nm"]
        )
