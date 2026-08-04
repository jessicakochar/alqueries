from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_active_learning
import run_cord_al
import run_tobacco3482_al


def test_unified_runner_parses_dataset_and_forwards_remaining_args():
    args, runner_args = run_active_learning.parse_args(
        [
            "--dataset",
            "cord",
            "--limit",
            "5",
            "--rounds",
            "1",
        ]
    )

    assert args.dataset == "cord"
    assert runner_args == ["--limit", "5", "--rounds", "1"]


def test_unified_runner_forwards_help_to_selected_dataset():
    args, runner_args = run_active_learning.parse_args(["--dataset", "cord", "--help"])

    assert args.dataset == "cord"
    assert runner_args == ["--help"]


def test_unified_runner_dispatches_to_selected_runner(monkeypatch):
    calls = []

    def fake_import_module(module_name):
        module = types.SimpleNamespace()
        module.main = lambda runner_args: calls.append((module_name, runner_args))
        return module

    monkeypatch.setattr(run_active_learning.importlib, "import_module", fake_import_module)

    run_active_learning.dispatch("tobacco3482", ["--limit", "10"])

    assert calls == [("run_tobacco3482_al", ["--limit", "10"])]


def test_existing_runners_accept_forwarded_args():
    cord_args = run_cord_al.parse_args(["--limit", "5", "--rounds", "1"])
    tobacco_args = run_tobacco3482_al.parse_args(["--limit", "5", "--rounds", "1"])

    assert cord_args.limit == 5
    assert cord_args.rounds == 1
    assert tobacco_args.limit == 5
    assert tobacco_args.rounds == 1
