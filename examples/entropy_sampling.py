from alqueries import get_strategy

from common import make_classification_engine

engine = make_classification_engine()
strategy = get_strategy("entropy")

picked = engine.query(strategy, n_samples=10)
print("entropy picks:", picked)
