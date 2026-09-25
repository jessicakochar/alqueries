from argparse import Namespace
from pathlib import Path
import sys

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_cord_al import (
    load_checkpoint,
    load_model_state_from_checkpoint,
    log_tensorboard_metrics,
    model_state_dict_to_cpu,
    save_run_history_csv,
    save_checkpoint,
    validate_checkpoint_labels,
)


class RecordingWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))


def test_save_and_load_checkpoint_round_state(tmp_path):
    checkpoint_path = tmp_path / "latest.pt"
    args = Namespace(strategy="token_entropy_sampling", rounds=2)
    metrics = {
        "epochs": 30,
        "train_loss": 2.5,
        "train_steps": 1.0,
        "eval_accuracy": 0.75,
        "eval_precision": 0.5,
        "eval_recall": 0.5,
        "eval_micro_f1": 0.5,
        "eval_split": "validation",
        "eval_steps": 2.0,
        "train_labeled_count": 1,
        "pre_query_unlabeled_count": 9,
        "post_query_labeled_count": 3,
        "post_query_unlabeled_count": 7,
        "selected_indices": [4, 5],
    }
    model_state_dict = {"classifier.weight": torch.ones(2, 2)}
    run_history = [{"round": 0, **metrics}]

    save_checkpoint(
        checkpoint_path,
        round_index=0,
        labeled_indices=np.array([0, 4, 5]),
        args=args,
        metrics=metrics,
        model_state_dict=model_state_dict,
        run_history=run_history,
    )
    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint["schema_version"] == 2
    assert checkpoint["round_index"] == 0
    assert checkpoint["labeled_indices"] == [0, 4, 5]
    assert checkpoint["args"]["strategy"] == "token_entropy_sampling"
    assert checkpoint["metrics"] == metrics
    assert checkpoint["run_history"] == run_history
    assert torch.equal(
        checkpoint["model_state_dict"]["classifier.weight"],
        model_state_dict["classifier.weight"],
    )


def test_log_tensorboard_metrics_writes_expected_scalars():
    writer = RecordingWriter()
    metrics = {
        "epochs": 30,
        "train_loss": 2.5,
        "train_steps": 1.0,
        "eval_accuracy": 0.75,
        "eval_precision": 0.5,
        "eval_recall": 0.5,
        "eval_micro_f1": 0.5,
        "eval_split": "validation",
        "eval_steps": 2.0,
        "train_labeled_count": 1,
        "pre_query_unlabeled_count": 9,
        "post_query_labeled_count": 3,
        "post_query_unlabeled_count": 7,
    }

    log_tensorboard_metrics(writer, metrics, round_index=2)

    assert writer.scalars == [
        ("train/loss", 2.5, 2),
        ("train/steps", 1.0, 2),
        ("eval/accuracy", 0.75, 2),
        ("eval/precision", 0.5, 2),
        ("eval/recall", 0.5, 2),
        ("eval/micro_f1", 0.5, 2),
        ("eval/steps", 2.0, 2),
        ("pool/train_labeled_count", 1, 2),
        ("pool/pre_query_unlabeled_count", 9, 2),
        ("pool/post_query_labeled_count", 3, 2),
        ("pool/post_query_unlabeled_count", 7, 2),
    ]


def test_save_run_history_csv_writes_eval_metrics(tmp_path):
    output_path = tmp_path / "results.csv"
    run_history = [
        {
            "round": 0,
            "epochs": 30,
            "train_loss": 2.5,
            "train_steps": 10.0,
            "eval_accuracy": 0.75,
            "eval_precision": 0.5,
            "eval_recall": 0.5,
            "eval_micro_f1": 0.5,
            "eval_split": "validation",
            "eval_steps": 3.0,
            "train_labeled_count": 10,
            "pre_query_unlabeled_count": 790,
            "post_query_labeled_count": 20,
            "post_query_unlabeled_count": 780,
            "selected_indices": [1, 2, 3],
        }
    ]

    save_run_history_csv(output_path, run_history)

    assert output_path.read_text().splitlines() == [
        "round,epochs,train_loss,train_steps,eval_accuracy,eval_precision,eval_recall,eval_micro_f1,eval_split,eval_steps,train_labeled_count,pre_query_unlabeled_count,post_query_labeled_count,post_query_unlabeled_count,selected_indices",
        '0,30,2.5,10.0,0.75,0.5,0.5,0.5,validation,3.0,10,790,20,780,"1, 2, 3"',
    ]


def test_model_state_helpers_save_cpu_weights_and_restore_model():
    source_model = torch.nn.Linear(2, 1)
    target_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source_model.weight.fill_(3.0)
        source_model.bias.fill_(1.0)
        target_model.weight.zero_()
        target_model.bias.zero_()

    model_state_dict = model_state_dict_to_cpu(source_model)
    checkpoint = {"model_state_dict": model_state_dict}

    assert all(not value.is_cuda for value in model_state_dict.values())
    assert load_model_state_from_checkpoint(target_model, checkpoint) is True
    assert torch.equal(target_model.weight, source_model.weight)
    assert torch.equal(target_model.bias, source_model.bias)


def test_load_model_state_from_checkpoint_handles_missing_weights():
    model = torch.nn.Linear(2, 1)

    assert load_model_state_from_checkpoint(model, {}) is False


def test_checkpoint_rejects_legacy_and_reordered_labels():
    labels = ["O", "B-MENU.NM", "I-MENU.NM"]
    for checkpoint in ({}, {"label_names": labels[::-1], "evaluation": "seqeval_entity_micro_bio"}):
        with pytest.raises(ValueError, match="Start a new BIO run"):
            validate_checkpoint_labels(checkpoint, labels)
    validate_checkpoint_labels(
        {"label_names": labels, "evaluation": "seqeval_entity_micro_bio"}, labels
    )
