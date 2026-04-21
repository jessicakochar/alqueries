import numpy as np

import alqueries.strategies.kmeans  # noqa: F401
from alqueries import get_strategy


def test_kmeans_sampling_picks_one_representative_per_cluster():
    strategy = get_strategy(
        "kmeans",
        pca_dim=None,
        cast_to_float16=False,
        kmeans_kwargs={"n_init": 10, "random_state": 0},
    )

    unlabeled = np.array([1, 2, 4, 5, 7, 8])

    # Full-dataset embeddings; strategy internally selects unlabeled rows.
    embeddings = np.zeros((9, 2), dtype=np.float32)
    embeddings[1] = [0.0, 0.0]
    embeddings[2] = [0.0, 2.0]
    embeddings[4] = [10.0, 10.0]
    embeddings[5] = [10.0, 12.0]
    embeddings[7] = [20.0, 20.0]
    embeddings[8] = [20.0, 22.0]

    embeddings[8] = [20.0, 22.0]

    picked = strategy.query(unlabeled_indices=unlabeled, n_samples=3, embeddings=embeddings)

    assert picked.shape == (3,)
    assert set(picked.tolist()) == {1, 4, 7}
