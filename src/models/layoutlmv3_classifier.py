from __future__ import annotations

import torch
from torch import nn

from models.common import classification_output


class LayoutLMv3Classifier(nn.Module):
    """
    LayoutLMv3 classifier for multimodal document inputs.

    It accepts text tokens, layout boxes, and page image tensors. The interface
    mirrors BertClassifier by returning loss, logits, and embeddings.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "microsoft/layoutlmv3-base",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from transformers import LayoutLMv3Model
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("Install `transformers` to use LayoutLMv3Classifier.") from exc

        self.layoutlmv3 = LayoutLMv3Model.from_pretrained(model_name)
        hidden_size = self.layoutlmv3.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        bbox: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if bbox is None:
            bbox = torch.zeros((*input_ids.shape, 4), dtype=torch.long, device=input_ids.device)

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        embeddings = outputs.pooler_output
        return classification_output(embeddings, self.dropout, self.classifier, labels)
