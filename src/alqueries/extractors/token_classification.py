from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alqueries.extractors.base import FeatureExtractor


class TokenClassificationFeatureExtractor(FeatureExtractor):
    """
    Extracts general token-classification features from a model.

    It returns token-level logits/probabilities/embeddings plus a valid-token
    mask. Document-level probabilities and embeddings are pooled convenience
    features for strategies that operate one score/vector per document.
    """

    def extract(self, loader: DataLoader) -> dict[str, np.ndarray | torch.Tensor]:
        self._model.to(self._device)
        self._model.eval()

        token_logits_chunks: list[torch.Tensor] = []
        token_embeddings_chunks: list[torch.Tensor] = []
        valid_token_mask_chunks: list[torch.Tensor] = []
        document_probs: list[torch.Tensor] = []
        document_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"]
                batch = _move_token_batch(batch, self._device)
                model_inputs = {
                    key: value
                    for key, value in batch.items()
                    if key != "labels"
                }
                outputs = self._model(**model_inputs, output_hidden_states=True)
                logits = outputs.logits.detach().cpu()
                token_embeddings = outputs.hidden_states[-1].detach().cpu()
                token_embeddings = token_embeddings[:, : labels.shape[1], :]
                valid_token_mask = labels.ne(-100)
                token_probs = F.softmax(logits, dim=-1)

                token_logits_chunks.append(logits)
                token_embeddings_chunks.append(token_embeddings)
                valid_token_mask_chunks.append(valid_token_mask)

                for row_index in range(token_probs.shape[0]):
                    row_mask = valid_token_mask[row_index]
                    row_probs = token_probs[row_index][row_mask]
                    row_embeddings = token_embeddings[row_index][row_mask]

                    if row_probs.numel() == 0:
                        row_probs = token_probs[row_index, :1]
                        row_embeddings = token_embeddings[row_index, :1]

                    document_probs.append(row_probs.mean(dim=0))
                    document_embeddings.append(row_embeddings.mean(dim=0))

        token_logits = torch.cat(token_logits_chunks, dim=0)
        token_embeddings = torch.cat(token_embeddings_chunks, dim=0)
        valid_token_mask = torch.cat(valid_token_mask_chunks, dim=0)

        return {
            "token_logits": token_logits,
            "token_probs": F.softmax(token_logits, dim=-1),
            "token_embeddings": token_embeddings,
            "valid_token_mask": valid_token_mask,
            "probs": torch.stack(document_probs, dim=0),
            "embeddings": torch.stack(document_embeddings, dim=0).numpy(),
        }


def _move_token_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key != "sample_index"
    }
