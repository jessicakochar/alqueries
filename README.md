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

## CORD evaluation

CORD uses the 61 BIO labels from the [official LayoutLMv3 CORD loader](https://github.com/microsoft/unilm/blob/master/layoutlmv3/layoutlmft/data/cord.py).
Each annotated line starts with `B-<category>`; subsequent words use `I-<category>`.
`other` words use `O`. CORD-v2 `menu.sub.*` categories are normalized to `MENU.SUB_*`.
The label vocabulary is fixed across dataset limits and train/validation/test splits.
Unknown categories raise an error rather than silently disappearing from evaluation.

After each round, `evaluate_layoutlmv3_token_classifier()` computes entity-level
micro precision, recall and F1 using `seqeval`, preserving receipt boundaries and
ignoring `-100` labels (special tokens, padding and continuation subwords).
It uses seqeval's default mode, matching the [official evaluation example](https://github.com/microsoft/unilm/blob/master/layoutlmv3/examples/run_funsd_cord.py),
not its optional strict IOB2 mode. `eval_accuracy` remains token accuracy.
The CSV and TensorBoard report `eval_precision`, `eval_recall` and `eval_micro_f1`;
the CSV also records `eval_split` and epochs per round.

Older CSV files report token macro F1 on category-only labels. Those scores cannot
be converted to entity micro F1 from the CSV. Old model heads also lack BIO boundary
labels, so start a new experiment without `--resume`, using separate output paths:

```bash
uv sync
uv run python run_active_learning.py --dataset cord \
  --rounds 1 --epochs 30 --batch-size 2 --initial-size 10 --query-size 10 \
  --checkpoint-dir checkpoints/cord_bio \
  --tensorboard-dir runs/cord_bio \
  --results-csv results/cord_bio.csv
```

Check a first round before extending the run. Resume a compatible BIO run using
`--resume checkpoints/cord_bio/latest.pt` with the same dataset, model and output paths.
The runner validates checkpoint label names and the evaluation scheme.

CORD writes `latest.pt` before training and after every completed epoch, using a
temporary file followed by replacement. Epoch checkpoints contain model and
optimizer state, random-number-generator state, loss totals, the labeled pool,
and completed-round history. Resume continues the same round from the next epoch;
an interrupted epoch is repeated. If training finished but evaluation/query did
not, resume reruns evaluation/query without repeating training. CSV results are
written after each completed round, so an epoch checkpoint can exist before the
first CSV row.

Each active-learning round starts from pretrained weights with a round-specific
seed, both in uninterrupted runs and after a completed-round resume. Model and
optimizer state are restored only when resuming inside a round. Keep training,
dataset, and evaluation settings unchanged when resuming; the total round target
may increase. Exact numeric reproducibility also depends on hardware and backend
determinism. Older checkpoints cannot recover progress within an interrupted
epoch or round. Only `latest.pt` includes the state needed for epoch resume;
the numbered round files contain round summaries.

CSV updates use temporary-file replacement and are rebuilt from checkpoint history
on resume if a previous CSV write was interrupted. Token entropy runs skip unused
hidden-state embeddings to reduce pool-extraction memory. A missing receipt image
raises an error when the real image processor is enabled.

This aligns the label scheme and metric, not the complete paper experiment.
Our default per-round evaluation uses `validation`; the official example evaluates
`test`. Keep test data held out for final reporting. Our preprocessing still uses
word boxes and truncates receipts to `--max-length` (default 256), while the official
example uses segment boxes and overflow chunks. Report these differences when
comparing results with the paper; the current metrics cover retained tokens only.
