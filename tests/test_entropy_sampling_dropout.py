import numpy as np
import torch
from alqueries.strategies.entropy_sampling_dropout import EntropySamplingDropout

def test_entropy_sampling_dropout_selects_most_uncertain():
    strategy = EntropySamplingDropout()
    probs = torch.tensor([
        [[0.90, 0.10], [0.50, 0.50], [0.55, 0.45], [0.95, 0.05]],
        [[0.90, 0.10], [0.50, 0.50], [0.55, 0.45], [0.95, 0.05]],
    ], dtype=torch.float32)  # shape: (T=2, N=4, C=2)

    unlabeled_indices = np.array([0, 1, 2, 3])

    selected = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=2,
        probs=probs,
    )

    assert len(selected) == 2
    assert set(selected.tolist()) == {1, 2}