from __future__ import annotations

import numpy as np
import torch

from alqueries.strategies.kcenter_greedy_safe import KCenterGreedy


def test_kcenter_greedy():

    strategy = KCenterGreedy()

    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [10.0, 10.0],
    ])

    labeled_indices = np.array([0])

    unlabeled_indices = np.array([1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        labeled_indices=labeled_indices,
        n_samples=2,
        embeddings=embeddings,
    )

    assert selected.tolist() == [3, 2]