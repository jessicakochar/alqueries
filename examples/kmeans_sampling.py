import alqueries.strategies.kmeans  # noqa: F401
from alqueries import get_strategy

from common import make_classification_engine

engine = make_classification_engine()
strategy = get_strategy(
    "kmeans",
    pca_dim=None,
    cast_to_float16=False,
    kmeans_kwargs={"n_init": 10, "random_state": 0},
)

picked = engine.query(strategy, n_samples=10)
print("kmeans picks:", picked)
