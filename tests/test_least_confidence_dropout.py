from __future__ import annotations

import numpy as np
import torch

from alqueries.strategies.least_confidence_dropout import (
    LeastConfidenceDropoutSampling,
)


def test_least_confidence_dropout():

    strategy = LeastConfidenceDropoutSampling()

    unlabeled_indices = np.array([0, 1, 2])

    mc_probs = torch.tensor([
        [
            [0.90, 0.10],
            [0.55, 0.45],
            [0.51, 0.49],
        ],
        [
            [0.85, 0.15],
            [0.50, 0.50],
            [0.52, 0.48],
        ],
    ])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=1,
        mc_probs=mc_probs,
    )

    assert selected.tolist() == [2]
