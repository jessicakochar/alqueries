import numpy as np
import torch
from alqueries import get_strategy


def test_margin_sampling_dropout_selects_smallest_mean_margin():
    strategy = get_strategy("margin_sampling_dropout")

    single = torch.tensor([
        [0.90, 0.10],
        [0.55, 0.45],
        [0.60, 0.40],
        [0.95, 0.05],
    ], dtype=torch.float32)
    probs = single.unsqueeze(0).expand(2, -1, -1)  # (T=2, N=4, C=2)

    unlabeled_indices = np.array([0, 1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        probs=probs,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {1, 2}  # the two smallest margins


# def test_margin_sampling_dropout():
#    probs = torch.tensor([
#       [
#       [0.9, 0.1],
#       [0.6, 0.4],
#       [0.51, 0.49]
#     ],
#     [
#        [0.85, 0.15],
#        [0.55, 0.45],
#        [0.52, 0.48],
#     ]
#     ])
#    selected = margin_sampling_dropout(probs, n_query=2)

#    assert selected[0].item() == 2
#    assert selected[1].item() == 1
#    assert len(selected) == 2
