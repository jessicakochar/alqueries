import numpy as np
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy

@register_strategy("entropy_sampling_dropout")
class EntropySamplingDropout(QueryStrategy):

    def query(self, unlabeled_indices, n_samples, *, probs, **_):
        mean_probs = probs.mean(dim=0)[unlabeled_indices]
        entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=1)
        return unlabeled_indices[entropy.argsort(descending=True)[:n_samples]]
