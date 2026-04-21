# alqueries

Tiny active-learning query strategies for classification pools.

## Install

```bash
uv add alqueries
```

## What you get

- `entropy`: uncertainty sampling from class probabilities
- `kmeans`: diverse sampling from embedding space
- `random`: seeded random baseline

## Minimal use

```python
import numpy as np
from alqueries import get_strategy

pool_indices = np.arange(100)

entropy = get_strategy("entropy")
picked = entropy.query(
	pool_indices=pool_indices,
	n_samples=10,
	probs=probs,  # torch.Tensor with shape (N, C)
)
```

For embeddings, use `kmeans` with `embeddings` shaped `(N, D)`.

## Run tests

```bash
uv run pytest -q
```
