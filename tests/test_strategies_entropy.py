import numpy as np
import torch

from alqueries import get_strategy


def test_entropy_sampling_selects_most_uncertain_unlabeled_points():
    strategy = get_strategy("entropy")

    # Full dataset probabilities (rows are absolute dataset indices).
    probs = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.99, 0.01],
        ],
        dtype=torch.float32,
    )
    unlabeled = np.array([1, 2, 3])

    picked = strategy.query(unlabeled_indices=unlabeled, n_samples=2, probs=probs)

    assert picked.shape == (2,)
    # Highest uncertainty in unlabeled set is index 2, then 3.
    assert set(picked.tolist()) == {2, 3}
