from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alqueries import QueryEngine, get_strategy
from alqueries.extractors import HuggingFaceClassificationFeatureExtractor
from alqueries.huggingface import (
    TextClassificationDataset,
    TextBatchCollator,
    evaluate_hf_text_classifier,
    load_tobacco3482_ocr,
    train_hf_text_classifier,
)
from models import BertClassifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERT active learning on Tobacco3482 OCR.")
    parser.add_argument("--strategy", default="entropy_sampling")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--initial-size", type=int, default=10)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mc-dropout-runs", type=int, default=5)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Persistent Hugging Face cache directory for dataset/model downloads.",
    )
    return parser.parse_args(argv)


def split_pool_test(dataset_size: int, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)
    test_size = max(1, int(dataset_size * test_ratio))
    return indices[test_size:], indices[:test_size]


def print_selected_samples(dataset: Subset, selected_indices: np.ndarray, max_print: int = 5) -> None:
    print("\nSelected samples:")
    for selected_index in selected_indices[:max_print]:
        item = dataset[int(selected_index)]
        preview = item["text"][:200].replace("\n", " ")
        print(f"- pool_index={int(selected_index)} label={item['label']} text={preview}...")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")

    cache_dir = None
    if args.cache_dir is not None:
        cache_path = Path(args.cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_path)
        print(f"Using cache dir: {cache_dir}")

    data = load_tobacco3482_ocr(split="train", limit=args.limit, cache_dir=cache_dir)
    full_dataset = TextClassificationDataset(data.texts, data.labels)
    pool_indices, test_indices = split_pool_test(len(full_dataset), args.test_ratio, args.seed)
    pool_dataset = Subset(full_dataset, pool_indices.tolist())
    test_dataset = Subset(full_dataset, test_indices.tolist())

    print(f"Loaded Tobacco3482 OCR samples: {len(full_dataset)}")
    print(f"Pool samples: {len(pool_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {len(data.label_names)}")
    print(f"Label names: {data.label_names}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    rng = np.random.default_rng(args.seed)
    initial_labeled = rng.choice(
        np.arange(len(pool_dataset)),
        size=min(args.initial_size, len(pool_dataset)),
        replace=False,
    )
    query_engine = QueryEngine(pool_dataset, labeled_indices=initial_labeled)
    if "features" not in inspect.signature(query_engine.query).parameters:
        raise RuntimeError(
            "This checkout is stale: QueryEngine.query() must support features=... . "
            "Pull/push the latest repo changes before running this script."
        )

    for round_index in range(args.rounds):
        print("\n" + "=" * 80)
        print(f"ACTIVE LEARNING ROUND {round_index}")
        print("=" * 80)

        model = BertClassifier(
            num_labels=len(data.label_names),
            model_name="bert-base-uncased",
            cache_dir=cache_dir,
        )
        train_metrics = train_hf_text_classifier(
            model,
            pool_dataset,
            query_engine.labeled_indices,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            max_length=args.max_length,
        )
        eval_metrics = evaluate_hf_text_classifier(
            model,
            test_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device,
            max_length=args.max_length,
        )

        print(f"Labeled samples: {len(query_engine.labeled_indices)}")
        print(f"Unlabeled samples: {len(query_engine.unlabeled_indices)}")
        print(f"Train loss: {train_metrics['train_loss']:.4f}")
        print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Macro F1: {eval_metrics['macro_f1']:.4f}")

        if len(query_engine.unlabeled_indices) == 0:
            print("No unlabeled samples left.")
            break

        feature_loader = DataLoader(
            pool_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=TextBatchCollator(tokenizer=tokenizer, max_length=args.max_length),
        )
        extractor = HuggingFaceClassificationFeatureExtractor(
            model=model,
            device=device,
            mc_dropout_runs=args.mc_dropout_runs,
        )
        features = extractor.extract(feature_loader)
        strategy = get_strategy(args.strategy)
        selected_indices = query_engine.query(
            strategy,
            n_samples=min(args.query_size, len(query_engine.unlabeled_indices)),
            features=features,
        )
        selected_indices = np.asarray(selected_indices, dtype=np.int64)

        print_selected_samples(pool_dataset, selected_indices)
        query_engine.add_labeled_indices(selected_indices)

    print("\nFinished active learning run.")


if __name__ == "__main__":
    main()
