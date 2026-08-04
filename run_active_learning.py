from __future__ import annotations

import argparse
import importlib


RUNNERS = {
    "cord": ("run_cord_al", "main"),
    "tobacco3482": ("run_tobacco3482_al", "main"),
    "tobacco": ("run_tobacco3482_al", "main"),
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run active learning for supported document datasets.",
        add_help=False,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(RUNNERS),
        help="Dataset pipeline to run.",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show unified help. Use --dataset DATASET -- --help for runner help.",
    )

    args, runner_args = parser.parse_known_args(argv)
    if args.help and args.dataset is None:
        parser.print_help()
        raise SystemExit(0)
    if args.dataset is None:
        parser.error("--dataset is required.")
    if args.help:
        runner_args.append("--help")
    return args, runner_args


def dispatch(dataset: str, runner_args: list[str]) -> None:
    module_name, function_name = RUNNERS[dataset]
    module = importlib.import_module(module_name)
    runner_main = getattr(module, function_name)
    runner_main(runner_args)


def main(argv: list[str] | None = None) -> None:
    args, runner_args = parse_args(argv)
    dispatch(args.dataset, runner_args)


if __name__ == "__main__":
    main()
