from alqueries import get_strategy

from common import make_tensor_pool
from alqueries import QueryEngine

engine = QueryEngine(make_tensor_pool(), labeled_indices=range(20))
strategy = get_strategy("random", seed=7)

picked = engine.query(strategy, n_samples=10)
print("random picks:", picked)
