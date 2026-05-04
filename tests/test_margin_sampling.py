# tests/test_margin_sampling.py
import numpy as np
import torch
from alqueries import get_strategy


def test_margin_sampling_selects_smallest_margin():
    strategy = get_strategy("margin_sampling")

    probs = torch.tensor([
        [0.70, 0.20, 0.10],
        [0.40, 0.35, 0.25],
        [0.60, 0.30, 0.10],
        [0.50, 0.45, 0.05],
        [0.80, 0.15, 0.05],
    ], dtype=torch.float32)

    unlabeled_indices = np.array([0, 1, 2, 3, 4])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        probs=probs,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {1, 3}  # the two smallest margins