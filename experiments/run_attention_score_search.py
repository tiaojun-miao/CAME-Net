"""CLI entrypoint for compact attention score search."""

from __future__ import annotations

import argparse
from typing import Sequence

from .attention_score_search import AttentionScoreSearchConfig, run_attention_score_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact robustness pilot for attention score variants.")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--class-protocol", choices=["full40", "small5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    return parser


def build_config_from_cli(argv: Sequence[str] | None = None) -> AttentionScoreSearchConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    kwargs = {}
    if args.class_protocol is not None:
        kwargs["class_protocol"] = args.class_protocol
    if args.methods is not None:
        kwargs["methods"] = tuple(args.methods)
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.num_points is not None:
        kwargs["num_points"] = args.num_points
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.hidden_dim is not None:
        kwargs["hidden_dim"] = args.hidden_dim
    if args.device is not None:
        kwargs["device"] = args.device
    if args.artifact_root is not None:
        kwargs["artifact_root"] = args.artifact_root
    if args.data_root is not None:
        kwargs["data_root"] = args.data_root
    return AttentionScoreSearchConfig(**kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    config = build_config_from_cli(argv)
    result = run_attention_score_search(config)
    print(f"Artifact directory: {result['artifact_dir']}")
    if result["results"]:
        best = result["results"][0]
        print(f"Best method: {best['method']}")
        print(f"Best mean shift accuracy: {best['mean_shift_accuracy']}")


if __name__ == "__main__":
    main()
