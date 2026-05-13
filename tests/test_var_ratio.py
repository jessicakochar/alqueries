import numpy as np
import torch
from alqueries import get_strategy


def test_var_ratio_selects_highest_disagreement():
    strategy = get_strategy("var_ratio")

    probs = torch.tensor([
        [0.90, 0.10],
        [0.55, 0.45],
        [0.52, 0.48],
        [0.95, 0.05],
    ], dtype=torch.float32)

    unlabeled_indices = np.array([0, 1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        probs=probs,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {1, 2}
