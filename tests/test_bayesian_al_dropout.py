# import torch
# from alqueries.strategies.bayesian_al_dropout import bayesian_al_dropout
# # from alqueries.strategies import bayesian_al_dropout

# def test_bayesian_dropout():
#     probs = torch.tensor([
#         [
#             [0.9, 0.1],
#             [0.6, 0.4],
#             [0.5, 0.5]
#         ],
#         [
#             [0.85, 0.15],
#             [0.55, 0.45],
#             [0.2, 0.8]
#         ]
#     ])

#     selected = bayesian_al_dropout(probs, n_query=2)
#     assert selected[0].item() == 2
#     assert len(selected) == 2

# tests/test_bayesian_al_dropout.py
import numpy as np
import torch
from alqueries import get_strategy


def test_bayesian_al_dropout_selects_highest_disagreement():
    strategy = get_strategy("bayesian_active_learning_disagreement_dropout")

    probs = torch.tensor([
        # T=0
        [[0.90, 0.10],   # idx 0
         [0.95, 0.05],   # idx 1
         [0.70, 0.30],   # idx 2
         [0.85, 0.15]],  # idx 3
        # T=1
        [[0.90, 0.10],   # idx 0  ← same as T=0
         [0.05, 0.95],   # idx 1  ← opposite of T=0
         [0.30, 0.70],   # idx 2  ← opposite of T=0
         [0.85, 0.15]],  # idx 3  ← same as T=0
    ], dtype=torch.float32)  # (T=2, N=4, C=2)

    unlabeled_indices = np.array([0, 1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        probs=probs,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {1, 2}  # highest inter-pass disagreement