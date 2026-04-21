import numpy as np

from alqueries import get_strategy


def test_random_is_seeded_and_without_replacement():
    pool_indices = np.arange(50, 60)

    strategy_a = get_strategy("random", seed=7)
    strategy_b = get_strategy("random", seed=7)

    picked_a = strategy_a.query(pool_indices=pool_indices, n_samples=5)
    picked_b = strategy_b.query(pool_indices=pool_indices, n_samples=5)

    assert picked_a.shape == (5,)
    assert np.array_equal(picked_a, picked_b)
    assert len(set(picked_a.tolist())) == 5
    assert set(picked_a.tolist()).issubset(set(pool_indices.tolist()))


def test_random_caps_requested_samples_to_pool_size():
    pool_indices = np.array([1, 2, 3])
    strategy = get_strategy("random", seed=0)

    picked = strategy.query(pool_indices=pool_indices, n_samples=10)

    assert picked.shape == (3,)
    assert set(picked.tolist()) == {1, 2, 3}
