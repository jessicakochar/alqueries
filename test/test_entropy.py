import numpy as np
import torch

from alqueries import get_strategy


def test_entropy_returns_highest_entropy_pool_indices():
    pool_indices = np.array([10, 11, 12, 13])
    probs = torch.tensor(
        [
            [1.0, 0.0],   # lowest entropy
            [0.5, 0.5],   # highest entropy
            [0.9, 0.1],
            [0.6, 0.4],
        ],
        dtype=torch.float32,
    )

    strategy = get_strategy("entropy")
    picked = strategy.query(pool_indices=pool_indices, n_samples=2, probs=probs)

    assert picked.shape == (2,)
    assert set(picked.tolist()) == {11, 13}
