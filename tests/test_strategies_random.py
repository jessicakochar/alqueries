import numpy as np

from alqueries import get_strategy


def test_random_sampling_is_seeded_and_in_unlabeled_pool():
    strategy_a = get_strategy("random", seed=7)
    strategy_b = get_strategy("random", seed=7)
    unlabeled = np.arange(10, 20)

    picked_a = strategy_a.query(unlabeled_indices=unlabeled, n_samples=5)
    picked_b = strategy_b.query(unlabeled_indices=unlabeled, n_samples=5)

    assert picked_a.shape == (5,)
    assert np.array_equal(picked_a, picked_b)
    assert len(set(picked_a.tolist())) == 5
    assert set(picked_a.tolist()).issubset(set(unlabeled.tolist()))


def test_random_sampling_caps_to_pool_size():
    strategy = get_strategy("random", seed=0)
    unlabeled = np.array([1, 2, 3])

    picked = strategy.query(unlabeled_indices=unlabeled, n_samples=10)

    assert picked.shape == (3,)
    assert set(picked.tolist()) == {1, 2, 3}
