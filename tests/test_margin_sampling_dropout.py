import numpy as np
import torch

from alqueries.strategies.margin_sampling_dropout import (
    MarginSamplingDropout,
)


def test_margin_sampling_dropout():

    strategy = MarginSamplingDropout()

    unlabeled_indices = np.array([0, 1, 2])

    mc_probs = torch.tensor([
        [
            [0.90, 0.10],
            [0.51, 0.49],
            [0.60, 0.40],
        ],
        [
            [0.88, 0.12],
            [0.52, 0.48],
            [0.61, 0.39],
        ],
    ])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=1,
        mc_probs=mc_probs,
    )

    assert selected.tolist() == [1]