from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alqueries import QueryEngine, get_strategy
from alqueries.extractors import TokenClassificationFeatureExtractor
from alqueries.huggingface import (
    create_layoutlmv3_token_classifier,
    evaluate_layoutlmv3_token_classifier,
    load_cord_token_classification,
    train_layoutlmv3_token_classifier,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CORD LayoutLMv3 token-classification active learning."
    )
    parser.add_argument("--strategy", default="token_entropy_sampling")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of CORD samples to load. Omit for the full dataset.",
    )
    parser.add_argument("--initial-size", type=int, default=1)
    parser.add_argument("--query-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/cord_bio")
    parser.add_argument("--tensorboard-dir", default="runs/cord_bio")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--full-labeled-loader",
        action="store_true",
        help="Deprecated; full labeled training is now the default.",
    )
    parser.add_argument(
        "--one-batch-smoke-test",
        action="store_true",
        help="Train one batch per epoch for quick debugging only.",
    )
    return parser.parse_args(argv)


def print_selected_receipts(dataset, selected_indices: np.ndarray, max_print: int = 3) -> None:
    print("\nSelected CORD receipts:")
    for selected_index in selected_indices[:max_print]:
        item = dataset[int(selected_index)]
        valid_tokens = int(item["labels"].ne(-100).sum().item())
        print(f"- pool_index={int(selected_index)} valid_token_labels={valid_tokens}")

def save_checkpoint(
    path,
    *,
    round_index,
    labeled_indices,
    args,
    metrics,
    model_state_dict=None,
    run_history=None,
    label_names=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64).tolist()
    torch.save(
        {
            "schema_version": 2,
            "label_names": label_names,
            "evaluation": "seqeval_entity_micro_bio",
            "round_index": round_index,
            "labeled_indices": labeled_indices,
            "args": vars(args),
            "metrics": metrics,
            "model_state_dict": model_state_dict,
            "run_history": run_history or [],
        },
        path,
    )


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_checkpoint_labels(checkpoint, label_names):
    if (
        checkpoint.get("label_names") != label_names
        or checkpoint.get("evaluation") != "seqeval_entity_micro_bio"
    ):
        raise ValueError(
            "Checkpoint uses an old or different CORD label/evaluation scheme. "
            "Start a new BIO run with separate checkpoint, CSV and TensorBoard paths; "
            "old category-only checkpoints cannot resume BIO training."
        )


def model_state_dict_to_cpu(model):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in model.state_dict().items()
    }


def load_model_state_from_checkpoint(model, checkpoint) -> bool:
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        return False
    model.load_state_dict(model_state_dict)
    return True


def save_run_history_csv(path, run_history):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "epochs",
        "train_loss",
        "train_steps",
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_micro_f1",
        "eval_split",
        "eval_steps",
        "train_labeled_count",
        "pre_query_unlabeled_count",
        "post_query_labeled_count",
        "post_query_unlabeled_count",
        "selected_indices",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in run_history:
            row = {key: record.get(key) for key in fieldnames}
            row["selected_indices"] = ", ".join(
                str(index) for index in record.get("selected_indices", [])
            )
            writer.writerow(row)


def log_tensorboard_metrics(writer, metrics, round_index):
    writer.add_scalar("train/loss", metrics["train_loss"], round_index)
    writer.add_scalar("train/steps", metrics["train_steps"], round_index)
    writer.add_scalar("eval/accuracy", metrics["eval_accuracy"], round_index)
    writer.add_scalar("eval/precision", metrics["eval_precision"], round_index)
    writer.add_scalar("eval/recall", metrics["eval_recall"], round_index)
    writer.add_scalar("eval/micro_f1", metrics["eval_micro_f1"], round_index)
    writer.add_scalar("eval/steps", metrics["eval_steps"], round_index)
    writer.add_scalar("pool/train_labeled_count", metrics["train_labeled_count"], round_index)
    writer.add_scalar("pool/pre_query_unlabeled_count", metrics["pre_query_unlabeled_count"], round_index)
    writer.add_scalar("pool/post_query_labeled_count", metrics["post_query_labeled_count"], round_index)
    writer.add_scalar("pool/post_query_unlabeled_count", metrics["post_query_unlabeled_count"], round_index)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")
    print("Images: enabled; using real CORD receipt pixel_values.")

    cache_dir = None
    if args.cache_dir is not None:
        cache_path = Path(args.cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_path)
        print(f"Using cache dir: {cache_dir}")

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
    )
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(
        args.model_name,
        apply_ocr=False,
        cache_dir=cache_dir,
    )
    cord = load_cord_token_classification(
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="train",
        limit=args.limit,
        max_length=args.max_length,
        cache_dir=cache_dir,
    )
    dataset = cord.dataset
    eval_cord = load_cord_token_classification(
        tokenizer=tokenizer,
        image_processor=image_processor,
        split=args.eval_split,
        limit=args.eval_limit,
        max_length=args.max_length,
        cache_dir=cache_dir,
        label_names=cord.label_names,
        label_to_id={label: index for index, label in enumerate(cord.label_names)},
    )
    eval_dataset = eval_cord.dataset
    print(f"Loaded CORD train samples: {len(dataset)}")
    print(f"Loaded CORD eval samples: {len(eval_dataset)}")
    print(f"Token labels: {len(cord.label_names)}")
    print(f"Label names: {cord.label_names}")

    # rng = np.random.default_rng(args.seed)
    # initial_labeled = rng.choice(
    #     np.arange(len(dataset)),
    #     size=min(args.initial_size, len(dataset)),
    #     replace=False,
    # )
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv = (
        Path(args.results_csv).expanduser().resolve()
        if args.results_csv is not None
        else checkpoint_dir / "cord_active_learning_results.csv"
    )
    tensorboard_dir = Path(args.tensorboard_dir).expanduser().resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard logs: {tensorboard_dir}")

    start_round = 0
    resume_checkpoint = None

    if args.resume:
        resume_checkpoint = load_checkpoint(Path(args.resume).expanduser().resolve())
        validate_checkpoint_labels(resume_checkpoint, cord.label_names)

        initial_labeled = np.asarray(resume_checkpoint["labeled_indices"], dtype=np.int64)
        start_round = resume_checkpoint["round_index"] + 1

        print(f"Resumed from checkpoint: {args.resume}")

    else:
        rng = np.random.default_rng(args.seed)

        initial_labeled = rng.choice(
            np.arange(len(dataset)),
            size=min(args.initial_size, len(dataset)),
            replace=False,
        )
    query_engine = QueryEngine(dataset, labeled_indices=initial_labeled)
    run_history = list(resume_checkpoint.get("run_history", [])) if resume_checkpoint else []
    last_checkpoint_path = None

    for round_index in range(start_round, args.rounds):
        print("\n" + "=" * 80)
        print(f"CORD ACTIVE LEARNING ROUND {round_index}")
        print("=" * 80)

        model = create_layoutlmv3_token_classifier(
            num_labels=len(cord.label_names),
            label_names=cord.label_names,
            model_name=args.model_name,
            cache_dir=cache_dir,
        )
        if resume_checkpoint is not None and round_index == start_round:
            if load_model_state_from_checkpoint(model, resume_checkpoint):
                print("Loaded model weights from checkpoint.")
            else:
                print("Checkpoint has no model weights; starting model from pretrained weights.")

        train_metrics = train_layoutlmv3_token_classifier(
            model,
            dataset,
            query_engine.labeled_indices,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            one_batch=args.one_batch_smoke_test,
        )
        print(f"Labeled receipts: {len(query_engine.labeled_indices)}")
        print(f"Unlabeled receipts: {len(query_engine.unlabeled_indices)}")
        print(f"Epochs: {args.epochs}")
        print(f"Train steps: {train_metrics['train_steps']:.0f}")
        print(f"Train loss: {train_metrics['train_loss']:.4f}")

        eval_metrics = evaluate_layoutlmv3_token_classifier(
            model,
            eval_dataset,
            label_names=cord.label_names,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"Eval steps: {eval_metrics['eval_steps']:.0f}")
        print(f"Eval accuracy: {eval_metrics['eval_accuracy']:.4f}")
        print(f"Eval entity precision: {eval_metrics['eval_precision']:.4f}")
        print(f"Eval entity recall: {eval_metrics['eval_recall']:.4f}")
        print(f"Eval entity micro F1 (seqeval): {eval_metrics['eval_micro_f1']:.4f}")

        train_labeled_count = len(query_engine.labeled_indices)
        pre_query_unlabeled_count = len(query_engine.unlabeled_indices)
        selected_indices = np.array([], dtype=np.int64)

        if pre_query_unlabeled_count == 0:
            print("No unlabeled receipts left.")
        else:
            feature_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )
            extractor = TokenClassificationFeatureExtractor(
                model=model,
                device=device,
            )
            features = extractor.extract(feature_loader)
            strategy = get_strategy(args.strategy)
            selected_indices = query_engine.query(
                strategy,
                n_samples=min(args.query_size, pre_query_unlabeled_count),
                features=features,
            )
            selected_indices = np.asarray(selected_indices, dtype=np.int64)
            print_selected_receipts(dataset, selected_indices)
            query_engine.add_labeled_indices(selected_indices)
        metrics = {
                "epochs": args.epochs,
                "train_loss": train_metrics["train_loss"],
                "train_steps": train_metrics["train_steps"],
                "eval_accuracy": eval_metrics["eval_accuracy"],
                "eval_precision": eval_metrics["eval_precision"],
                "eval_recall": eval_metrics["eval_recall"],
                "eval_micro_f1": eval_metrics["eval_micro_f1"],
                "eval_split": args.eval_split,
                "eval_steps": eval_metrics["eval_steps"],
                "train_labeled_count": train_labeled_count,
                "pre_query_unlabeled_count": pre_query_unlabeled_count,
                "post_query_labeled_count": len(query_engine.labeled_indices),
                "post_query_unlabeled_count": len(query_engine.unlabeled_indices),
                "selected_indices": selected_indices.tolist(),
            }
        run_history.append({
            "round": round_index,
            **metrics,
        })
        log_tensorboard_metrics(writer, metrics, round_index)
        save_run_history_csv(results_csv, run_history)

        checkpoint_path = checkpoint_dir / f"round_{round_index}.pt"
        save_checkpoint(
                checkpoint_path,
                round_index=round_index,
                labeled_indices=query_engine.labeled_indices,
                args=args,
                metrics=metrics,
                run_history=run_history,
                label_names=cord.label_names,
            )

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            round_index=round_index,
            labeled_indices=query_engine.labeled_indices,
            args=args,
            metrics=metrics,
            model_state_dict=model_state_dict_to_cpu(model),
            run_history=run_history,
            label_names=cord.label_names,
            )

        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Saved results CSV: {results_csv}")
        last_checkpoint_path = checkpoint_path
        if pre_query_unlabeled_count == 0:
            break

    writer.flush()
    writer.close()

    print("\n" + "=" * 80)
    print("CORD ACTIVE LEARNING SUMMARY")
    print("=" * 80)
    print(f"Rounds completed: {len(run_history)}")
    print(f"Final labeled receipts: {len(query_engine.labeled_indices)}")
    print(f"Final unlabeled receipts: {len(query_engine.unlabeled_indices)}")
    print(f"Epochs per round: {args.epochs}")
    if last_checkpoint_path is not None:
        print(f"Last checkpoint: {last_checkpoint_path}")

    for record in run_history:
        train_labeled_count = record.get("train_labeled_count", record.get("labeled_count"))
        post_query_labeled_count = record.get(
            "post_query_labeled_count",
            record.get("labeled_count"),
        )
        post_query_unlabeled_count = record.get(
            "post_query_unlabeled_count",
            record.get("unlabeled_count"),
        )
        print(
            f"Round {record['round']}: "
            f"loss={record['train_loss']:.4f}, "
            f"epochs={record.get('epochs', 'n/a')}, "
            f"eval_accuracy={record.get('eval_accuracy', 0.0):.4f}, "
            f"eval_micro_f1={record['eval_micro_f1']:.4f}, "
            f"train_labeled={train_labeled_count}, "
            f"post_query_labeled={post_query_labeled_count}, "
            f"post_query_unlabeled={post_query_unlabeled_count}, "
            f"selected={record['selected_indices']}"
        )
    print("\nFinished CORD active learning run.")


if __name__ == "__main__":
    main()
