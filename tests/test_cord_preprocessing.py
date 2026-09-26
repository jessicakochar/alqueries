import json

import torch
import pytest
from PIL import Image

from alqueries.huggingface.cord import (
    CordTokenClassificationDataset,
    DEFAULT_IMAGE_SIZE,
    CORD_LABEL_NAMES,
    extract_cord_words_labels_boxes,
    _align_word_labels,
)


class FakeTokenizerOutput(dict):
    def __init__(self, word_ids):
        super().__init__(
            input_ids=torch.tensor([[101, 2000, 2001, 102, 0]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
            bbox=torch.zeros(1, 5, 4, dtype=torch.long),
            labels=torch.tensor([[-100, 0, 1, -100, -100]], dtype=torch.long),
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class FakeTokenizer:
    def __call__(self, words, **kwargs):
        assert words == ["Coffee", "Total"]
        assert kwargs["boxes"] == [[100, 100, 200, 200], [300, 300, 400, 400]]
        assert kwargs["word_labels"] == [0, 1]
        assert kwargs["padding"] == "max_length"
        return FakeTokenizerOutput([None, 0, 1, None, None])


class FakeImageProcessor:
    def __init__(self):
        self.seen_modes = []

    def __call__(self, image, **kwargs):
        self.seen_modes.append(image.mode)
        assert kwargs["return_tensors"] == "pt"
        return {
            "pixel_values": torch.full(
                (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                0.5,
                dtype=torch.float32,
            )
        }


def _raw_cord_sample():
    return {
        "ground_truth": json.dumps(
            {
                "valid_line": [
                    {
                        "category": "menu",
                        "words": [
                            {
                                "text": "Coffee",
                                "quad": {
                                    "x1": 10,
                                    "y1": 10,
                                    "x2": 20,
                                    "y2": 10,
                                    "x3": 20,
                                    "y3": 20,
                                    "x4": 10,
                                    "y4": 20,
                                },
                            }
                        ],
                    },
                    {
                        "category": "total",
                        "words": [
                            {
                                "text": "Total",
                                "quad": {
                                    "x1": 30,
                                    "y1": 30,
                                    "x2": 40,
                                    "y2": 30,
                                    "x3": 40,
                                    "y3": 40,
                                    "x4": 30,
                                    "y4": 40,
                                },
                            }
                        ],
                    },
                ]
            }
        ),
        "image": Image.new("L", (100, 100)),
    }


def test_cord_preprocessing_outputs_matching_sequence_lengths():
    dataset = CordTokenClassificationDataset(
        [_raw_cord_sample()],
        tokenizer=FakeTokenizer(),
        label_to_id={"B-MENU": 0, "B-TOTAL": 1},
        max_length=5,
    )

    item = dataset[0]
    sequence_length = item["input_ids"].shape[0]

    assert item["attention_mask"].shape[0] == sequence_length
    assert item["bbox"].shape[0] == sequence_length
    assert item["labels"].shape[0] == sequence_length
    assert item["pixel_values"].shape == (
        3,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )


def test_cord_preprocessing_uses_real_image_processor_when_available():
    image_processor = FakeImageProcessor()
    dataset = CordTokenClassificationDataset(
        [_raw_cord_sample()],
        tokenizer=FakeTokenizer(),
        image_processor=image_processor,
        label_to_id={"B-MENU": 0, "B-TOTAL": 1},
        max_length=5,
    )

    item = dataset[0]

    assert image_processor.seen_modes == ["RGB"]
    assert torch.all(item["pixel_values"] == 0.5)


def test_cord_bio_boundaries_other_and_v2_category_aliases():
    sample = {"ground_truth": json.dumps({"valid_line": [
        {"category": "menu.nm", "words": [{"text": " "}, {"text": "Hot"}, {"text": "Coffee"}]},
        {"category": "menu.nm", "words": [{"text": "Tea"}]},
        {"category": "other", "words": [{"text": "Thank"}, {"text": "you"}]},
        {"category": "menu.sub.nm", "words": [{"text": "Extra"}, {"text": "milk"}]},
    ]})}
    labels = [label for _, label, _ in extract_cord_words_labels_boxes(sample)]
    assert labels == ["B-MENU.NM", "I-MENU.NM", "B-MENU.NM", "O", "O", "B-MENU.SUB_NM", "I-MENU.SUB_NM"]
    assert set(labels) <= set(CORD_LABEL_NAMES)
    assert len(CORD_LABEL_NAMES) == len(set(CORD_LABEL_NAMES)) == 61


def test_cord_unknown_labels_are_not_silently_ignored():
    dataset = CordTokenClassificationDataset(
        [_raw_cord_sample()], tokenizer=FakeTokenizer(), label_to_id={"O": 0}
    )
    with pytest.raises(ValueError, match="Unknown CORD BIO labels"):
        dataset[0]


def test_cord_subword_alignment_keeps_only_first_token():
    assert _align_word_labels([None, 0, 0, 1, None], [1, 2]) == [-100, 1, -100, 2, -100]


def test_missing_real_image_is_rejected():
    from alqueries.huggingface.cord import _pixel_values
    with pytest.raises(ValueError, match="image is missing"):
        _pixel_values({}, FakeImageProcessor())
