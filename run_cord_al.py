from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alqueries import QueryEngine, get_strategy
from alqueries.huggingface import (
    create_layoutlmv3_token_classifier,
    load_cord_token_classification,
    predict_layoutlmv3_token_features,
    train_layoutlmv3_token_classifier,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small CORD LayoutLMv3 token-classification active-learning smoke test."
    )
    parser.add_argument("--strategy", default="token_entropy_sampling")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--initial-size", type=int, default=1)
    parser.add_argument("--query-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--checkpoint-dir", default="checkpoints/cord")
    parser.add_argument("--tensorboard-dir", default="runs/cord")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--full-labeled-loader",
        action="store_true",
        help="Use all labeled batches. Default is one batch for the 10-epoch smoke test.",
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
):
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64).tolist()
    torch.save(
        {
            "schema_version": 1,
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


def log_tensorboard_metrics(writer, metrics, round_index):
    writer.add_scalar("train/loss", metrics["train_loss"], round_index)
    writer.add_scalar("train/steps", metrics["train_steps"], round_index)
    writer.add_scalar("pool/labeled_count", metrics["labeled_count"], round_index)
    writer.add_scalar("pool/unlabeled_count", metrics["unlabeled_count"], round_index)

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
    print(f"Loaded CORD samples: {len(dataset)}")
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
    tensorboard_dir = Path(args.tensorboard_dir).expanduser().resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard logs: {tensorboard_dir}")

    start_round = 0
    resume_checkpoint = None

    if args.resume:
        resume_checkpoint = load_checkpoint(Path(args.resume).expanduser().resolve())

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
            one_batch=not args.full_labeled_loader,
        )
        print(f"Labeled receipts: {len(query_engine.labeled_indices)}")
        print(f"Unlabeled receipts: {len(query_engine.unlabeled_indices)}")
        print(f"Train steps: {train_metrics['train_steps']:.0f}")
        print(f"Train loss: {train_metrics['train_loss']:.4f}")

        if len(query_engine.unlabeled_indices) == 0:
            print("No unlabeled receipts left.")
            break

        features = predict_layoutlmv3_token_features(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
        )
        strategy = get_strategy(args.strategy)
        selected_indices = query_engine.query(
            strategy,
            n_samples=min(args.query_size, len(query_engine.unlabeled_indices)),
            features=features,
        )
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        print_selected_receipts(dataset, selected_indices)
        query_engine.add_labeled_indices(selected_indices)
        metrics = {
                "train_loss": train_metrics["train_loss"],
                "train_steps": train_metrics["train_steps"],
                "labeled_count": len(query_engine.labeled_indices),
                "unlabeled_count": len(query_engine.unlabeled_indices),
                "selected_indices": selected_indices.tolist(),
            }
        run_history.append({
            "round": round_index,
            **metrics,
        })
        log_tensorboard_metrics(writer, metrics, round_index)

        checkpoint_path = checkpoint_dir / f"round_{round_index}.pt"
        save_checkpoint(
                checkpoint_path,
                round_index=round_index,
                labeled_indices=query_engine.labeled_indices,
                args=args,
                metrics=metrics,
                run_history=run_history,
            )

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            round_index=round_index,
            labeled_indices=query_engine.labeled_indices,
            args=args,
            metrics=metrics,
            model_state_dict=model_state_dict_to_cpu(model),
            run_history=run_history,
            )

        print(f"Saved checkpoint: {checkpoint_path}")
        last_checkpoint_path = checkpoint_path

    writer.flush()
    writer.close()

    print("\n" + "=" * 80)
    print("CORD ACTIVE LEARNING SUMMARY")
    print("=" * 80)
    print(f"Rounds completed: {len(run_history)}")
    print(f"Final labeled receipts: {len(query_engine.labeled_indices)}")
    print(f"Final unlabeled receipts: {len(query_engine.unlabeled_indices)}")
    if last_checkpoint_path is not None:
        print(f"Last checkpoint: {last_checkpoint_path}")

    for record in run_history:
        print(
            f"Round {record['round']}: "
            f"loss={record['train_loss']:.4f}, "
            f"labeled={record['labeled_count']}, "
            f"unlabeled={record['unlabeled_count']}, "
            f"selected={record['selected_indices']}"
        )
    print("\nFinished CORD active learning smoke run.")


if __name__ == "__main__":
    main()
