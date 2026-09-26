from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, Subset


IGNORE_INDEX = -100
DEFAULT_IMAGE_SIZE = 224
CORD_ENTITY_TYPES = (
    "MENU.NM", "MENU.NUM", "MENU.UNITPRICE", "MENU.CNT", "MENU.DISCOUNTPRICE",
    "MENU.PRICE", "MENU.ITEMSUBTOTAL", "MENU.VATYN", "MENU.ETC", "MENU.SUB_NM",
    "MENU.SUB_UNITPRICE", "MENU.SUB_CNT", "MENU.SUB_PRICE", "MENU.SUB_ETC",
    "VOID_MENU.NM", "VOID_MENU.PRICE", "SUB_TOTAL.SUBTOTAL_PRICE",
    "SUB_TOTAL.DISCOUNT_PRICE", "SUB_TOTAL.SERVICE_PRICE", "SUB_TOTAL.OTHERSVC_PRICE",
    "SUB_TOTAL.TAX_PRICE", "SUB_TOTAL.ETC", "TOTAL.TOTAL_PRICE", "TOTAL.TOTAL_ETC",
    "TOTAL.CASHPRICE", "TOTAL.CHANGEPRICE", "TOTAL.CREDITCARDPRICE", "TOTAL.EMONEYPRICE",
    "TOTAL.MENUTYPE_CNT", "TOTAL.MENUQTY_CNT",
)
CORD_LABEL_NAMES = ["O"] + [
    f"{prefix}-{entity}" for prefix in ("B", "I") for entity in CORD_ENTITY_TYPES
]


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
    label_to_id: dict[str, int] | None = None,
    label_names: list[str] | None = None,
) -> CordData:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("Install `datasets` to load CORD.") from exc

    raw_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if limit is not None:
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))

    if label_names is None:
        label_names = list(CORD_LABEL_NAMES)
    if label_to_id is None:
        label_to_id = {label: index for index, label in enumerate(label_names)}
    if label_to_id != {label: index for index, label in enumerate(label_names)}:
        raise ValueError("label_to_id must match the ordering of label_names.")
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
        category = line.get("category", "other").upper().replace("MENU.SUB.", "MENU.SUB_")
        first_word = True
        for word in line.get("words", []):
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            label = "O" if category == "OTHER" else f"{'B' if first_word else 'I'}-{category}"
            items.append((text, label, _word_bbox(word, width=width, height=height)))
            first_word = False

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
        unknown_labels = {item[1] for item in words_labels_boxes} - self.label_to_id.keys()
        if unknown_labels:
            raise ValueError(f"Unknown CORD BIO labels: {sorted(unknown_labels)}")
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
    resume_state: dict | None = None,
    checkpoint_callback: Any | None = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    if epochs <= 0 or batch_size <= 0 or len(labeled_indices) == 0:
        raise ValueError("Training requires positive epochs, batch size, and a nonempty labeled pool.")
    loader = DataLoader(
        Subset(dataset, labeled_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_loss = 0.0
    steps = 0
    start_epoch = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = resume_state["completed_epochs"]
        if not 0 <= start_epoch <= epochs:
            raise ValueError("Checkpoint completed_epochs must be between zero and epochs.")
        total_loss = resume_state["total_loss"]
        steps = resume_state["steps"]
        random.setstate(resume_state["python_rng"])
        np.random.set_state(resume_state["numpy_rng"])
        torch.set_rng_state(resume_state["torch_rng"].cpu())
        if torch.device(device).type == "cuda" and resume_state["cuda_rng"]:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])

    def save_progress(completed_epochs):
        if checkpoint_callback is not None:
            checkpoint_callback({
                "completed_epochs": completed_epochs,
                "optimizer_state_dict": optimizer.state_dict(),
                "total_loss": total_loss,
                "steps": steps,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            })

    if resume_state is None:
        save_progress(0)

    for epoch in range(start_epoch, epochs):
        print(f"[Train] Epoch {epoch + 1}/{epochs} started | batches={len(loader)}", flush=True)
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
        save_progress(epoch + 1)
        print(f"[Train] Epoch {epoch + 1}/{epochs} complete | cumulative loss={total_loss / max(steps, 1):.4f}", flush=True)

    return {"train_loss": total_loss / max(steps, 1), "train_steps": float(steps)}


def evaluate_layoutlmv3_token_classifier(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    label_names: list[str],
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    if not label_names or any(
        label != "O" and not label.startswith(("B-", "I-")) for label in label_names
    ):
        raise ValueError("CORD evaluation requires BIO label names.")
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: list[list[str]] = []
    all_labels: list[list[str]] = []
    steps = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_cord_batch(batch, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            if outputs.logits.shape[-1] != len(label_names):
                raise ValueError("Model output classes must match the BIO label names.")
            preds = outputs.logits.argmax(dim=-1)
            for receipt_preds, receipt_labels in zip(preds, labels):
                valid_mask = receipt_labels.ne(IGNORE_INDEX)
                if valid_mask.any():
                    all_preds.append([
                        label_names[index] for index in receipt_preds[valid_mask].cpu().tolist()
                    ])
                    all_labels.append([
                        label_names[index] for index in receipt_labels[valid_mask].cpu().tolist()
                    ])
            steps += 1

    if not all_labels:
        raise ValueError("CORD evaluation contains no valid labeled tokens.")

    return {
        "eval_accuracy": accuracy_score(all_labels, all_preds),
        "eval_precision": precision_score(all_labels, all_preds, average="micro", zero_division=0),
        "eval_recall": recall_score(all_labels, all_preds, average="micro", zero_division=0),
        "eval_micro_f1": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "eval_steps": float(steps),
    }


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
        raise ValueError("CORD receipt image is missing; multimodal training requires real images.")

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
