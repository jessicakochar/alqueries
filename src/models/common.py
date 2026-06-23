from __future__ import annotations

import torch
from torch import nn


def classification_output(
    embeddings: torch.Tensor,
    dropout: nn.Dropout,
    classifier: nn.Linear,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    logits = classifier(dropout(embeddings))
    loss = None
    if labels is not None:
        loss = nn.CrossEntropyLoss()(logits, labels)
    return {"loss": loss, "logits": logits, "embeddings": embeddings}
