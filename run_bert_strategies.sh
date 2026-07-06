#!/bin/bash

set -e

CACHE_DIR="${ALQUERIES_CACHE_DIR:-.cache/huggingface}"
mkdir -p "$CACHE_DIR"

for strategy in random_sampling entropy_sampling least_confidence margin_sampling
do
  echo "============================================"
  echo "Running strategy: $strategy"
  echo "============================================"

  PYTHONPATH=src python run_tobacco3482_al.py \
    --strategy "$strategy" \
    --limit 100 \
    --initial-size 10 \
    --query-size 10 \
    --rounds 1 \
    --epochs 1 \
    --batch-size 4 \
    --cache-dir "$CACHE_DIR"
done
