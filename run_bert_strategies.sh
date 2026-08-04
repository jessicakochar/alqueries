#!/bin/bash

set -e

CACHE_DIR="${ALQUERIES_CACHE_DIR:-.cache/huggingface}"
mkdir -p "$CACHE_DIR"
for strategy in \
random_sampling \
entropy_sampling \
least_confidence \
margin_sampling \
entropy_sampling_dropout \
least_confidence_dropout \
margin_sampling_dropout \
bald_dropout \
mean_std \
var_ratio \
kmeans \
kcenter_greedy_safe

do
  echo "============================================"
  echo "Running strategy: $strategy"
  echo "============================================"

  PYTHONPATH=src python run_tobacco3482_al.py \
  --strategy "$strategy" \
  --limit 500 \
  --initial-size 50 \
  --query-size 50 \
  --rounds 5 \
  --epochs 3 \
  --batch-size 8 \
  --cache-dir "$CACHE_DIR"
donen