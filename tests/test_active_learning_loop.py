from __future__ import annotations

import numpy as np
import torch

from alqueries.training import ActiveLearningLoop, ActiveLearningLoopConfig


class TinyDataset:
    def __len__(self):
        return 5


def test_active_learning_loop_trains_queries_and_updates_indices():
    train_calls: list[list[int]] = [] #store labeled indices that train_fn receeives

    def model_builder():
        return object() #dummy model

    def train_fn(_model, _dataset, labeled_indices):
        train_calls.append(labeled_indices.tolist())
        return {"train_loss": 0.5}

    def predict_fn(_model, _dataset):
        return {
            "probs": torch.tensor(
                [
                    [0.99, 0.01],
                    [0.50, 0.50],
                    [0.95, 0.05],
                    [0.51, 0.49],
                    [0.90, 0.10],
                ],
                dtype=torch.float32,
            )
        }

    loop = ActiveLearningLoop(
        TinyDataset(),
        model_builder=model_builder,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=ActiveLearningLoopConfig(
            initial_size=1,
            query_size=2,
            rounds=2,
            strategy_name="least_confidence",
        ),
        initial_indices=np.array([0]),
    )

    results = loop.run()

    assert len(results) == 2
    assert results[0].selected_indices.tolist() == [1, 3]
    assert train_calls == [[0], [0, 1, 3]]
