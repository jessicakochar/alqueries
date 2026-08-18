from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


IGNORE_INDEX = -100
DEFAULT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class CordData:
    dataset: "CordTokenClassificationDataset"
    label_names: list[str]


def load_cord_token_classification(
    *,
    tokenizer: Any,
    image_processor: Any | None = None,
    split: str = "train",
    dataset_name: str = "naver-clova-ix/cord-v2",
    limit: int | None = None,
    max_length: int = 256,
    cache_dir: str | None = None,
) -> CordData:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("Install `datasets` to load CORD.") from exc

    raw_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if limit is not None:
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))

    label_names = collect_cord_label_names(raw_dataset)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    return CordData(
        dataset=CordTokenClassificationDataset(
            raw_dataset,
            tokenizer=tokenizer,
            image_processor=image_processor,
            label_to_id=label_to_id,
            max_length=max_length,
        ),
        label_names=label_names,
    )


def collect_cord_label_names(raw_dataset) -> list[str]:
    labels: set[str] = set()
    for sample in raw_dataset:
        for _word, label, _box in extract_cord_words_labels_boxes(sample):
            labels.add(label)
    return sorted(labels)


def extract_cord_words_labels_boxes(sample: dict[str, Any]) -> list[tuple[str, str, list[int]]]:
    parsed = json.loads(sample["ground_truth"])
    width, height = _image_size(sample)
    items: list[tuple[str, str, list[int]]] = []

    for line in parsed.get("valid_line", []):
        label = line.get("category", "other")
        for word in line.get("words", []):
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            items.append((text, label, _word_bbox(word, width=width, height=height)))

    return items


class CordTokenClassificationDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        *,
        tokenizer: Any,
        image_processor: Any | None = None,
        label_to_id: dict[str, int],
        max_length: int = 256,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | list[str]]:
        sample = self.raw_dataset[int(index)]
        words_labels_boxes = extract_cord_words_labels_boxes(sample)
        words = [item[0] for item in words_labels_boxes]
        word_labels = [self.label_to_id[item[1]] for item in words_labels_boxes]
        boxes = [item[2] for item in words_labels_boxes]

        encoded = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        if "labels" in encoded:
            token_labels = encoded["labels"].squeeze(0).tolist()
        else:
            token_labels = _align_word_labels(word_ids, word_labels)

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
            if key != "labels"
        }
        seq_len = item["input_ids"].shape[0]
        assert item["attention_mask"].shape[0] == seq_len
        assert item["bbox"].shape[0] == seq_len
        assert len(token_labels) == seq_len

        item["labels"] = torch.tensor(token_labels, dtype=torch.long)
        item["pixel_values"] = _pixel_values(sample, self.image_processor)
        item["sample_index"] = int(index)
        return item


def train_layoutlmv3_token_classifier(
    model: torch.nn.Module,
    dataset: Dataset,
    labeled_indices: np.ndarray,
    *,
    batch_size: int = 1,
    epochs: int = 10,
    lr: float = 5e-5,
    device: str | torch.device = "cpu",
    one_batch: bool = True,
) -> dict[str, float]:
    model.to(device)
    model.train()
    loader = DataLoader(
        Subset(dataset, labeled_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_loss = 0.0
    steps = 0

    for _epoch in range(epochs):
        for batch in loader:
            batch = _move_cord_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
            if one_batch:
                break

    return {"train_loss": total_loss / max(steps, 1), "train_steps": float(steps)}


def create_layoutlmv3_token_classifier(
    *,
    num_labels: int,
    label_names: list[str],
    model_name: str = "microsoft/layoutlmv3-base",
    cache_dir: str | None = None,
) -> torch.nn.Module:
    from transformers import LayoutLMv3ForTokenClassification

    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in id2label.items()}
    return LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )


def _align_word_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    labels: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(word_labels[word_id])
        previous_word_id = word_id
    return labels


def _word_bbox(word: dict[str, Any], *, width: int, height: int) -> list[int]:
    quad = word.get("quad") or {}
    if quad:
        xs = [quad.get(key, 0) for key in ("x1", "x2", "x3", "x4")]
        ys = [quad.get(key, 0) for key in ("y1", "y2", "y3", "y4")]
        box = [min(xs), min(ys), max(xs), max(ys)]
    else:
        box = word.get("bbox") or word.get("box") or [0, 0, 0, 0]
    return _normalize_box(box, width=width, height=height)


def _normalize_box(box: list[float], *, width: int, height: int) -> list[int]:
    width = max(width, 1)
    height = max(height, 1)
    x0, y0, x1, y1 = box[:4]
    normalized = [
        int(1000 * float(x0) / width),
        int(1000 * float(y0) / height),
        int(1000 * float(x1) / width),
        int(1000 * float(y1) / height),
    ]
    return [max(0, min(1000, value)) for value in normalized]


def _image_size(sample: dict[str, Any]) -> tuple[int, int]:
    image = sample.get("image")
    if image is not None and hasattr(image, "size"):
        return int(image.size[0]), int(image.size[1])
    return 1000, 1000


def _pixel_values(sample: dict[str, Any], image_processor: Any | None) -> torch.Tensor:
    if image_processor is None:
        return torch.zeros(
            3,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_IMAGE_SIZE,
            dtype=torch.float32,
        )

    image = sample.get("image")
    if image is None:
        return torch.zeros(
            3,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_IMAGE_SIZE,
            dtype=torch.float32,
        )

    if hasattr(image, "convert"):
        image = image.convert("RGB")

    encoded_image = image_processor(image, return_tensors="pt")
    pixel_values = encoded_image["pixel_values"]
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.squeeze(0).to(dtype=torch.float32)
    return torch.tensor(pixel_values[0], dtype=torch.float32)


def _move_cord_batch(batch: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key != "sample_index"
    }
