import numpy as np

from alqueries import get_strategy


def test_kmeans_picks_one_representative_per_cluster():
    pool_indices = np.array([100, 101, 200, 201, 300, 301])
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [10.0, 10.0],
            [10.0, 12.0],
            [20.0, 20.0],
            [20.0, 22.0],
        ],
        dtype=np.float32,
    )

    strategy = get_strategy(
        "kmeans",
        pca_dim=None,
        cast_to_float16=False,
        kmeans_kwargs={"n_init": 10, "random_state": 0},
    )
    picked = strategy.query(pool_indices=pool_indices, n_samples=3, embeddings=embeddings)

    assert picked.shape == (3,)
    assert set(picked.tolist()) == {100, 200, 300}
