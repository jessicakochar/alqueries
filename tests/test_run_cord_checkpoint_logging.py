from argparse import Namespace
from pathlib import Path
import sys

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
    save_checkpoint,
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
        "train_loss": 2.5,
        "train_steps": 1.0,
        "labeled_count": 3,
        "unlabeled_count": 7,
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

    assert checkpoint["schema_version"] == 1
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
        "train_loss": 2.5,
        "train_steps": 1.0,
        "labeled_count": 3,
        "unlabeled_count": 7,
    }

    log_tensorboard_metrics(writer, metrics, round_index=2)

    assert writer.scalars == [
        ("train/loss", 2.5, 2),
        ("train/steps", 1.0, 2),
        ("pool/labeled_count", 3, 2),
        ("pool/unlabeled_count", 7, 2),
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
