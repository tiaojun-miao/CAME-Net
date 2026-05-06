"""User entrypoint for lightweight comparison and ablation experiments."""

from __future__ import annotations

import argparse
from typing import Sequence

from .comparison_baselines import list_comparison_methods
from .comparison_experiment import ComparisonExperimentConfig, run_comparison_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight comparison or ablation experiments.")
    parser.add_argument("--method", choices=list_comparison_methods(), default="came")
    parser.add_argument("--class-protocol", choices=["full40", "small5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--equiv-loss-weight", type=float, default=None)
    parser.add_argument("--aux-loss-weight", type=float, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    return parser


def build_config_from_cli(argv: Sequence[str] | None = None) -> ComparisonExperimentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_kwargs = {"method": args.method}
    if args.class_protocol is not None:
        config_kwargs["class_protocol"] = args.class_protocol

    if args.epochs is not None:
        config_kwargs["num_epochs"] = args.epochs
    if args.learning_rate is not None:
        config_kwargs["learning_rate"] = args.learning_rate
    if args.equiv_loss_weight is not None:
        config_kwargs["equiv_loss_weight"] = args.equiv_loss_weight
    if args.aux_loss_weight is not None:
        config_kwargs["aux_loss_weight"] = args.aux_loss_weight
    if args.num_points is not None:
        config_kwargs["num_points"] = args.num_points
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.hidden_dim is not None:
        config_kwargs["hidden_dim"] = args.hidden_dim
    if args.image_size is not None:
        config_kwargs["image_size"] = args.image_size
    if args.device is not None:
        config_kwargs["device"] = args.device
    if args.artifact_root is not None:
        config_kwargs["artifact_root"] = args.artifact_root
    if args.data_root is not None:
        config_kwargs["data_root"] = args.data_root

    return ComparisonExperimentConfig(**config_kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    config = build_config_from_cli(argv)
    result = run_comparison_experiment(config)
    metrics = result["metrics"]

    print(f"Method: {result['method']}")
    print(f"Artifact directory: {result['artifact_dir']}")
    print(f"Overall accuracy: {metrics.get('overall_accuracy', 'n/a')}")
    print(f"Mean class accuracy: {metrics.get('mean_class_accuracy', 'n/a')}")
    print(f"Parameter count: {metrics.get('parameter_count', 'n/a')}")


if __name__ == "__main__":
    main()
