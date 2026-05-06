from __future__ import annotations

import argparse

from .scannet_multimodal_experiment import ScanNetMultimodalConfig, run_scannet_multimodal_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ScanNet scene-level multimodal CAME experiment.")
    parser.add_argument("--data-root", required=True, help="Path to ScanNet root containing scans/")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num-points", type=int, default=256, help="Number of sampled mesh points per scene")
    parser.add_argument("--max-frames", type=int, default=3, help="Number of RGB frames per scene")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=32, help="CAME hidden dim")
    parser.add_argument("--artifact-root", default="artifacts/scannet_multimodal", help="Artifact root")
    args = parser.parse_args()

    config = ScanNetMultimodalConfig(
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        num_points=args.num_points,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    result = run_scannet_multimodal_experiment(config)
    print(f"Artifact directory: {result['artifact_dir']}")
    print(f"Micro-F1: {result['metrics']['micro_f1']:.4f}")
    print(f"Macro-F1: {result['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

