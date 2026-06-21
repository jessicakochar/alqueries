from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from alqueries.pool import QueryEngine
from alqueries.registry import get_strategy


ModelBuilder = Callable[[], Any] #New model instance
TrainFunction = Callable[[Any, Any, np.ndarray], Mapping[str, float]] #return trainig metrics-loss,accuracy etc
PredictFunction = Callable[[Any, Any], Mapping[str, Any]] #returs features for query_strategies- logits,probs, embeddings etc


@dataclass(frozen=True)
class ActiveLearningLoopConfig:
    initial_size: int
    query_size: int
    rounds: int
    strategy_name: str = "entropy_sampling"
    seed: int = 0
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveLearningRoundResult:
    round_index: int
    labeled_indices: np.ndarray
    unlabeled_indices: np.ndarray
    selected_indices: np.ndarray
    train_metrics: Mapping[str, float]


class ActiveLearningLoop:

    def __init__(
        self,
        dataset: Any,
        *,
        model_builder: ModelBuilder,
        train_fn: TrainFunction,
        predict_fn: PredictFunction,
        config: ActiveLearningLoopConfig,
        initial_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.dataset = dataset
        self.model_builder = model_builder
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        full_indices = np.arange(len(dataset), dtype=np.int64)
        if initial_indices is None:
            initial_count = min(config.initial_size, len(full_indices))
            labeled_indices = self.rng.choice(full_indices, size=initial_count, replace=False)
        else:
            labeled_indices = np.unique(np.asarray(initial_indices, dtype=np.int64))

        self.query_engine = QueryEngine(dataset, labeled_indices=np.sort(labeled_indices))


# The main loop running n times: train, predict, query, update indices, repeat.
    def run(self) -> list[ActiveLearningRoundResult]:
        results: list[ActiveLearningRoundResult] = []

        for round_index in range(self.config.rounds):
            model = self.model_builder()
            train_metrics = self.train_fn(model, self.dataset, self.query_engine.labeled_indices)
            features = dict(self.predict_fn(model, self.dataset))

            unlabeled_indices = self.query_engine.unlabeled_indices
            if len(unlabeled_indices) == 0:
                selected_indices = np.array([], dtype=np.int64)
            else:
                strategy = get_strategy(self.config.strategy_name, **self.config.strategy_kwargs)
                selected_indices = self.query_engine.query(
                    strategy,
                    n_samples=min(self.config.query_size, len(unlabeled_indices)),
                    features=features,
                )
                selected_indices = np.asarray(selected_indices, dtype=np.int64)

            results.append(
                ActiveLearningRoundResult(
                    round_index=round_index,
                    labeled_indices=self.query_engine.labeled_indices,
                    unlabeled_indices=unlabeled_indices,
                    selected_indices=selected_indices.copy(),
                    train_metrics=train_metrics,
                )
            )

            if len(selected_indices) == 0:
                break

            self.query_engine.add_labeled_indices(selected_indices)

        return results
