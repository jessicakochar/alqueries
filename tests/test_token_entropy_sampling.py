import numpy as np
import pytest
import torch

from alqueries import get_strategy


def test_token_entropy_sampling_selects_highest_mean_uncertainty():
    strategy = get_strategy("token_entropy_sampling")
    unlabeled = np.array([1, 3, 4])
    token_probs = torch.tensor(
        [
            [[0.99, 0.01]],
            [[0.50, 0.50]],
            [[0.95, 0.05]],
            [[0.80, 0.20]],
            [[0.55, 0.45]],
        ],
        dtype=torch.float32,
    )
    valid_token_mask = torch.ones(5, 1, dtype=torch.bool)

    selected = strategy.query(
        unlabeled_indices=unlabeled,
        n_samples=2,
        token_probs=token_probs,
        valid_token_mask=valid_token_mask,
    )

    assert selected.tolist() == [1, 4]


def test_token_entropy_sampling_requires_dataset_level_token_probs():
    strategy = get_strategy("token_entropy_sampling")

    with pytest.raises(ValueError, match="one row per dataset sample"):
        strategy.query(
            unlabeled_indices=np.array([2]),
            n_samples=1,
            token_probs=torch.full((2, 1, 2), 0.5),
            valid_token_mask=torch.ones(2, 1, dtype=torch.bool),
        )


def test_token_entropy_sampling_returns_empty_for_empty_unlabeled_pool():
    strategy = get_strategy("token_entropy_sampling")

    selected = strategy.query(
        unlabeled_indices=np.array([], dtype=np.int64),
        n_samples=1,
        token_probs=torch.full((2, 1, 2), 0.5),
        valid_token_mask=torch.ones(2, 1, dtype=torch.bool),
    )

    assert selected.tolist() == []
    assert selected.dtype == np.int64


def test_token_entropy_sampling_validates_n_samples():
    strategy = get_strategy("token_entropy_sampling")
    unlabeled = np.array([0, 1])
    token_probs = torch.full((2, 1, 2), 0.5)
    valid_token_mask = torch.ones(2, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="greater than zero"):
        strategy.query(
            unlabeled_indices=unlabeled,
            n_samples=0,
            token_probs=token_probs,
            valid_token_mask=valid_token_mask,
        )

    with pytest.raises(ValueError, match="cannot exceed"):
        strategy.query(
            unlabeled_indices=unlabeled,
            n_samples=3,
            token_probs=token_probs,
            valid_token_mask=valid_token_mask,
        )
