"""User entrypoint for the small ModelNet40 experiment."""

from __future__ import annotations

from .small_modelnet_experiment import SmallExperimentConfig, run_small_experiment


def main() -> None:
    config = SmallExperimentConfig()
    result = run_small_experiment(config)
    metrics = result["metrics"]

    print(f"Artifact directory: {result['artifact_dir']}")
    print(f"Overall accuracy: {metrics.get('overall_accuracy', 'n/a')}")
    print(f"Mean class accuracy: {metrics.get('mean_class_accuracy', 'n/a')}")
    print(f"Parameter count: {metrics.get('parameter_count', 'n/a')}")


if __name__ == "__main__":
    main()
