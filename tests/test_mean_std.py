from __future__ import annotations

import numpy as np
import torch

from alqueries.strategies.mean_std import MeanStd


def test_mean_std():

    strategy = MeanStd()

    unlabeled_indices = np.array([0, 1, 2, 3])

    mc_probs = torch.tensor([
        # T = 0
        [
            [0.90, 0.10],   # idx 0
            [0.95, 0.05],   # idx 1
            [0.70, 0.30],   # idx 2
            [0.85, 0.15],   # idx 3
        ],

        # T = 1
        [
            [0.90, 0.10],   # idx 0 (same)
            [0.05, 0.95],   # idx 1 (huge disagreement)
            [0.30, 0.70],   # idx 2 (medium disagreement)
            [0.85, 0.15],   # idx 3 (same)
        ],
    ], dtype=torch.float32)

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        mc_probs=mc_probs,
    )

    assert selected.tolist() == [1, 2]