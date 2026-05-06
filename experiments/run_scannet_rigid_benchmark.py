from __future__ import annotations

import argparse

from .scannet_rigid_benchmark import ScanNetRigidBenchmarkConfig, run_scannet_rigid_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ScanNet multimodal rigid benchmark.")
    parser.add_argument("--method", required=True, help="Benchmark method name")
    parser.add_argument("--data-root", required=True, help="Path to ScanNet root containing scans/")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--num-points", type=int, default=256, help="Number of sampled points per scene")
    parser.add_argument("--max-frames", type=int, default=3, help="Number of RGB frames per scene")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--top-k-labels", type=int, default=12, help="Top-K frequent labels retained in the benchmark vocabulary")
    parser.add_argument("--use-blind-holdout", action="store_true", help="Use blind holdout scenes as the test split")
    parser.add_argument("--artifact-root", default="artifacts/scannet_rigid_benchmark", help="Artifact output root")
    args = parser.parse_args()

    config = ScanNetRigidBenchmarkConfig(
        method=args.method,
        data_root=args.data_root,
        device=args.device,
        num_epochs=args.epochs,
        num_points=args.num_points,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        top_k_labels=args.top_k_labels,
        use_blind_holdout=args.use_blind_holdout,
        artifact_root=args.artifact_root,
    )
    result = run_scannet_rigid_benchmark(config)
    print(f"Artifact directory: {result['artifact_dir']}")
    print(f"Mean rigid micro-F1: {result['metrics']['mean_rigid_micro_f1']:.4f}")
    print(f"Prediction drift: {result['metrics']['prediction_drift']:.6f}")


if __name__ == "__main__":
    main()
