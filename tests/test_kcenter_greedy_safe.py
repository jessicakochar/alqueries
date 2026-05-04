# tests/test_kcenter_greedy_safe.py
import numpy as np
import torch
from alqueries import get_strategy


def test_kcenter_greedy_safe_selects_furthest_from_labelled():
    strategy = get_strategy("kcenter_greedy_safe")

    embeddings = torch.tensor([
        [0.0,  0.0],
        [1.0,  1.0],
        [10.0, 10.0],
        [12.0, 10.0],
        [15.0, 15.0],
    ], dtype=torch.float32)

    labeled_mask  = np.array([True, False, False, False, False])
    unlabeled_indices = np.array([1, 2, 3, 4])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        embeddings=embeddings,
        labeled_mask=labeled_mask,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {2, 4}  # the two most spread-out points


def test_kcenter_greedy_safe_cold_start_no_labelled_points():
    strategy = get_strategy("kcenter_greedy_safe")

    # No labelled points at all — cold start.
    # Seeds from idx 0 (first unlabelled), then picks the furthest point.
    embeddings = torch.tensor([
        [0.0,  0.0],   # idx 0 — seed for cold start
        [1.0,  0.0],   # idx 1 — close to seed
        [20.0, 0.0],   # idx 2 — furthest from seed ← must be selected
    ], dtype=torch.float32)

    labeled_mask      = np.array([False, False, False])
    unlabeled_indices = np.array([0, 1, 2])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=1,
        embeddings=embeddings,
        labeled_mask=labeled_mask,
    )

    assert len(selected) == 1
    assert selected[0] == 2


def test_kcenter_greedy_safe_no_duplicate_selections():
    strategy = get_strategy("kcenter_greedy_safe")

    embeddings = torch.tensor([
        [0.0, 0.0],
        [5.0, 0.0],
        [5.0, 5.0],
        [0.0, 5.0],
    ], dtype=torch.float32)

    labeled_mask      = np.array([True, False, False, False])
    unlabeled_indices = np.array([1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=3,
        embeddings=embeddings,
        labeled_mask=labeled_mask,
    )

    assert len(selected) == 3
    assert len(set(selected.tolist())) == 3