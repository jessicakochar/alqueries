import numpy as np
import torch
from alqueries import get_strategy


def test_var_ratio_selects_highest_disagreement():
    strategy = get_strategy("var_ratio")

    probs = torch.tensor([
        # T=0
        [[0.90, 0.10],
         [0.80, 0.20],
         [0.75, 0.25],
         [0.95, 0.05]],
        # T=1
        [[0.90, 0.10],
         [0.10, 0.90],
         [0.20, 0.80],
         [0.85, 0.15]],
    ], dtype=torch.float32)

    unlabeled_indices = np.array([0, 1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        probs=probs,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {1, 2}  # highest variation ratio