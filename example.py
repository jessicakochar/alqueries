import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from alqueries import QueryEngine
from alqueries.extractors.classification import ClassificationFeatureExtractor

N_POOL, N_FEATURES, N_CLASSES = 100, 16, 5
dataset = TensorDataset(
	torch.randn(N_POOL, N_FEATURES),
	torch.randint(0, N_CLASSES, (N_POOL,)),
)

model = nn.Sequential(
	nn.Linear(N_FEATURES, 32),
	nn.ReLU(),
	nn.Linear(32, N_CLASSES),
)
extractor = ClassificationFeatureExtractor(model=model, device="cpu", input_key=0)

# First 20 are labeled, remaining are unlabeled by default.
engine = QueryEngine(dataset, labeled_indices=np.arange(20), extractor=extractor)

picked_entropy = engine.query("entropy", n_samples=10)
picked_random_labeled = engine.query("random", n_samples=5, subset="labeled", seed=7)

# For k-center-like strategies, engine.query() always provides split metadata
# (labeled/unlabeled/full indices and labeled_mask). If extractor is set,
# features are extracted on full data and passed in kwargs.

print("entropy (unlabeled):", picked_entropy)
print("random  (labeled):  ", picked_random_labeled)