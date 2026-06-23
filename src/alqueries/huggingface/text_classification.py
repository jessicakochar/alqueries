from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset


class TextClassificationDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"text": self.texts[index], "label": self.labels[index]}


@dataclass
class TextBatchCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return encoded


def train_hf_text_classifier(
    model: torch.nn.Module,
    dataset: Dataset,
    labeled_indices: np.ndarray,
    *,
    tokenizer: Any,
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 2e-5,
    device: str | torch.device = "cpu",
    max_length: int = 512,
) -> dict[str, float]:
    if len(labeled_indices) == 0:
        raise ValueError("At least one labeled sample is required for training.")

    model.to(device)
    model.train()
    loader = DataLoader(
        Subset(dataset, labeled_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TextBatchCollator(tokenizer=tokenizer, max_length=max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_loss = 0.0
    total_steps = 0
    for _ in range(epochs):
        for batch in loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_steps += 1

    return {"train_loss": total_loss / max(total_steps, 1)}


def predict_hf_text_classifier(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    tokenizer: Any,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    max_length: int = 512,
    mc_dropout_runs: int = 10,
) -> dict[str, torch.Tensor | np.ndarray]:
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TextBatchCollator(tokenizer=tokenizer, max_length=max_length),
    )

    logits, embeddings = _predict_once(model, loader, device=device, train_mode=False)
    mc_logits = torch.stack(
        [
            _predict_once(model, loader, device=device, train_mode=True)[0]
            for _ in range(mc_dropout_runs)
        ],
        dim=0,
    )

    return {
        "logits": logits,
        "probs": F.softmax(logits, dim=1),
        "embeddings": embeddings.numpy(),
        "mc_logits": mc_logits,
        "mc_probs": F.softmax(mc_logits, dim=2),
    }


def evaluate_hf_text_classifier(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    tokenizer: Any,
    indices: np.ndarray | None = None,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    max_length: int = 512,
) -> dict[str, float]:
    model.to(device)
    model.eval()

    eval_dataset = dataset
    if indices is not None:
        eval_dataset = Subset(dataset, indices.tolist())

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TextBatchCollator(tokenizer=tokenizer, max_length=max_length),
    )

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs["logits"].argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def _predict_once(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device,
    train_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train(train_mode)
    logits_chunks: list[torch.Tensor] = []
    embedding_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch.pop("labels")
            outputs = model(**batch, labels=labels)
            logits_chunks.append(outputs["logits"].detach().cpu())
            embedding_chunks.append(outputs["embeddings"].detach().cpu())

    return torch.cat(logits_chunks, dim=0), torch.cat(embedding_chunks, dim=0)


def _move_batch(batch: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
