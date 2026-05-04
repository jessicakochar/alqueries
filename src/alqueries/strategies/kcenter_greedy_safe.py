import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("kcenter_greedy_safe")
class kcenter_greedy_safe(QueryStrategy):

    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        embeddings: torch.Tensor,
        labeled_mask: np.ndarray,
        **_,
    ) -> np.ndarray:
        # Work with a boolean tensor on the same device as embeddings.
        labelled_mask = torch.tensor(labeled_mask, dtype=torch.bool,
                                     device=embeddings.device)
        labelled_indices   = torch.where(labelled_mask)[0]
        unlabelled_indices = torch.where(~labelled_mask)[0]

        num = embeddings.shape[0]
        min_distances = torch.full((num,), float("inf"), device=embeddings.device)

        # Step 1 — initialise min_distances from existing labelled points.
        if len(labelled_indices) > 0:
            for idx in labelled_indices:
                dist = torch.norm(embeddings - embeddings[idx], dim=1)
                min_distances = torch.minimum(min_distances, dist)
        else:
            # Cold-start: no labelled points yet — seed from the first
            # unlabelled point so greedy selection has somewhere to start.
            first_idx = unlabelled_indices[0]
            min_distances = torch.norm(embeddings - embeddings[first_idx], dim=1)

        # Step 2 — greedily pick the furthest unlabelled point each step.
        selected = []
        for _ in range(n_samples):
            # Restrict to the current unlabelled pool.
            pool_distances = min_distances[unlabelled_indices]
            chosen_pos     = torch.argmax(pool_distances)
            chosen_idx     = unlabelled_indices[chosen_pos]
            selected.append(chosen_idx.item())

            # Update min_distances to account for the newly selected center.
            new_dist   = torch.norm(embeddings - embeddings[chosen_idx], dim=1)
            min_distances = torch.minimum(min_distances, new_dist)

            # Remove chosen_idx from the pool for the next iteration.
            keep = torch.ones(len(unlabelled_indices), dtype=torch.bool,
                              device=embeddings.device)
            keep[chosen_pos] = False
            unlabelled_indices = unlabelled_indices[keep]

        return np.array(selected, dtype=np.int64)