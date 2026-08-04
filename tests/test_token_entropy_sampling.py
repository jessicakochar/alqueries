import numpy as np
import pytest

from alqueries import get_strategy


def test_token_entropy_sampling_selects_highest_mean_uncertainty():
    strategy = get_strategy("token_entropy_sampling")
    unlabeled = np.array([1, 3, 4])
    mean_token_uncertainty = np.array([0.0, 0.8, 0.1, 0.5, 0.9])

    selected = strategy.query(
        unlabeled_indices=unlabeled,
        n_samples=2,
        mean_token_uncertainty=mean_token_uncertainty,
    )

    assert selected.tolist() == [4, 1]


def test_token_entropy_sampling_requires_dataset_level_scores():
    strategy = get_strategy("token_entropy_sampling")

    with pytest.raises(ValueError, match="one score per dataset sample"):
        strategy.query(
            unlabeled_indices=np.array([2]),
            n_samples=1,
            mean_token_uncertainty=np.array([0.1, 0.2]),
        )


def test_token_entropy_sampling_returns_empty_for_empty_unlabeled_pool():
    strategy = get_strategy("token_entropy_sampling")

    selected = strategy.query(
        unlabeled_indices=np.array([], dtype=np.int64),
        n_samples=1,
        mean_token_uncertainty=np.array([0.1, 0.2]),
    )

    assert selected.tolist() == []
    assert selected.dtype == np.int64


def test_token_entropy_sampling_validates_n_samples():
    strategy = get_strategy("token_entropy_sampling")
    unlabeled = np.array([0, 1])
    mean_token_uncertainty = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="greater than zero"):
        strategy.query(
            unlabeled_indices=unlabeled,
            n_samples=0,
            mean_token_uncertainty=mean_token_uncertainty,
        )

    with pytest.raises(ValueError, match="cannot exceed"):
        strategy.query(
            unlabeled_indices=unlabeled,
            n_samples=3,
            mean_token_uncertainty=mean_token_uncertainty,
        )
