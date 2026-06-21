import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("kcenter_greedy_safe")
@register_strategy("kcenter_greedy")
class KCenterGreedy(QueryStrategy):
    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        embeddings: torch.Tensor,
        labeled_indices: np.ndarray,
        **_,
    ) -> np.ndarray:
        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        unlabeled_embeddings = embeddings[unlabeled_indices]
        labeled_embeddings = embeddings[labeled_indices]
        if len(labeled_embeddings) == 0:
            return unlabeled_indices[: min(n_samples, len(unlabeled_indices))]
        distances = torch.cdist(unlabeled_embeddings, labeled_embeddings)
        min_distances = distances.min(dim=1).values
        selected: list[int] = []
        for _ in range(min(n_samples, len(unlabeled_indices))):
            farthest_idx = torch.argmax(min_distances).item()
            selected.append(farthest_idx)
            new_center = unlabeled_embeddings[farthest_idx].unsqueeze(0)
            new_distances = torch.cdist(unlabeled_embeddings, new_center).squeeze(1)
            min_distances = torch.minimum(min_distances, new_distances)
            min_distances[farthest_idx] = -torch.inf
        return np.asarray(unlabeled_indices[selected])
