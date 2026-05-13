from __future__ import annotations

import numpy as np
import torch

from alqueries.strategies.entropy_sampling_dropout import (
    EntropySamplingDropout,
)


def test_entropy_sampling_dropout():

    strategy = EntropySamplingDropout()

    unlabeled_indices = np.array([0, 1, 2])

    mc_probs = torch.tensor([
        [
            [0.99, 0.01],
            [0.50, 0.50],
            [0.70, 0.30],
        ],
        [
            [0.98, 0.02],
            [0.45, 0.55],
            [0.65, 0.35],
        ],
    ])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=1,
        mc_probs=mc_probs,
    )

    assert selected.tolist() == [1]