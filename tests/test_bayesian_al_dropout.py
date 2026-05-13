from __future__ import annotations

import numpy as np
import torch

from alqueries.strategies.bayesian_al_dropout import (
    BayesianALDropout,
)


def test_bayesian_active_learning_disagreement_dropout():

    strategy = BayesianALDropout()

    unlabeled_indices = np.array([0, 1])

    mc_probs = torch.tensor([
        [
            [0.99, 0.01],
            [0.50, 0.50],
        ],
        [
            [0.99, 0.01],
            [0.10, 0.90],
        ],
    ])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=1,
        mc_probs=mc_probs,
    )

    assert selected.tolist() == [1]