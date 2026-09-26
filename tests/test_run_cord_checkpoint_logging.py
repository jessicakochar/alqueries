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

    assert checkpoint["schema_version"] == 3
    assert checkpoint["phase"] == "round_complete"
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


def test_epoch_resume_matches_uninterrupted_training(tmp_path):
    from types import SimpleNamespace
    from alqueries.huggingface.cord import train_layoutlmv3_token_classifier

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(2, 2)

        def forward(self, input_ids, labels):
            logits = self.classifier(self.dropout(input_ids))
            return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels))

    dataset = [
        {"input_ids": torch.tensor([float(index), 1.0]), "labels": torch.tensor(index % 2)}
        for index in range(6)
    ]
    settings = dict(batch_size=2, epochs=3, lr=0.01, one_batch=False)
    torch.manual_seed(17)
    full_model = TinyModel()
    expected = train_layoutlmv3_token_classifier(full_model, dataset, np.arange(6), **settings)

    torch.manual_seed(17)
    interrupted_model = TinyModel()
    path = tmp_path / "latest.pt"

    def interrupt(state):
        save_checkpoint(
            path, round_index=0, labeled_indices=np.arange(6),
            args=Namespace(**settings), metrics={},
            model_state_dict=model_state_dict_to_cpu(interrupted_model),
            phase="training", training_state=state,
        )
        if state["completed_epochs"] == 1:
            raise RuntimeError("Simulated disconnect")

    with pytest.raises(RuntimeError, match="Simulated disconnect"):
        train_layoutlmv3_token_classifier(
            interrupted_model, dataset, np.arange(6), checkpoint_callback=interrupt, **settings
        )
    checkpoint = load_checkpoint(path)
    assert checkpoint["training_state"]["completed_epochs"] == 1
    restored_model = TinyModel()
    load_model_state_from_checkpoint(restored_model, checkpoint)
    actual = train_layoutlmv3_token_classifier(
        restored_model, dataset, np.arange(6),
        resume_state=checkpoint["training_state"], **settings
    )
    assert actual == expected
    for key, value in full_model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[key])
    assert not path.with_suffix(".pt.tmp").exists()


def test_runner_round_boundary_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import run_cord_al as runner

    dataset = [{"labels": torch.tensor([0])} for _ in range(4)]
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    for component in (runner.LayoutLMv3TokenizerFast, runner.LayoutLMv3ImageProcessor):
        monkeypatch.setattr(component, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_cord_token_classification", lambda **kwargs: SimpleNamespace(dataset=dataset, label_names=["O"]))
    monkeypatch.setattr(runner, "create_layoutlmv3_token_classifier", lambda **kwargs: torch.nn.Linear(2, 1))
    monkeypatch.setattr(runner, "TokenClassificationFeatureExtractor", lambda **kwargs: SimpleNamespace(extract=lambda loader, **options: {}))
    monkeypatch.setattr(runner, "get_strategy", lambda name: SimpleNamespace(query=lambda unlabeled_indices, n_samples, **kwargs: unlabeled_indices[:n_samples]))
    monkeypatch.setattr(runner, "evaluate_layoutlmv3_token_classifier", lambda *args, **kwargs: dict(eval_accuracy=1.0, eval_precision=1.0, eval_recall=1.0, eval_micro_f1=1.0, eval_steps=1.0))

    def train(model, *args, **kwargs):
        with torch.no_grad():
            model.weight.add_(torch.rand_like(model.weight))
        return {"train_loss": float(model.weight.detach().sum()), "train_steps": 1.0}

    monkeypatch.setattr(runner, "train_layoutlmv3_token_classifier", train)

    def arguments(folder, rounds):
        return ["--rounds", str(rounds), "--checkpoint-dir", str(folder), "--tensorboard-dir", str(folder / "logs")]

    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    runner.main(arguments(full_dir, 2))
    runner.main(arguments(resumed_dir, 1))
    (resumed_dir / "cord_active_learning_results.csv").unlink()
    runner.main(arguments(resumed_dir, 2) + ["--resume", str(resumed_dir / "latest.pt")])
    expected = load_checkpoint(full_dir / "latest.pt")
    actual = load_checkpoint(resumed_dir / "latest.pt")
    assert (resumed_dir / "cord_active_learning_results.csv").read_text() == (full_dir / "cord_active_learning_results.csv").read_text()
    assert expected["run_history"] == actual["run_history"]
    assert expected["labeled_indices"] == actual["labeled_indices"]
    for key, value in expected["model_state_dict"].items():
        assert torch.equal(value, actual["model_state_dict"][key])


@pytest.mark.parametrize("option", ["--epochs", "--rounds", "--batch-size", "--initial-size", "--query-size", "--limit"])
def test_invalid_run_sizes_fail_before_loading_models(option):
    from run_cord_al import parse_args
    with pytest.raises(SystemExit):
        parse_args([option, "0"])


def test_failed_checkpoint_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import run_cord_al as runner
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous checkpoint")

    def failed_save(payload, destination):
        destination.write_bytes(b"incomplete")
        raise OSError("Disk full")

    monkeypatch.setattr(runner.torch, "save", failed_save)
    with pytest.raises(OSError, match="Disk full"):
        save_checkpoint(path, round_index=0, labeled_indices=[0], args=Namespace(), metrics={})
    assert path.read_bytes() == b"previous checkpoint"
