from __future__ import annotations

import argparse

from .scannet_paper_visualizations import ScanNetPaperFigureConfig, generate_scannet_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-grade ScanNet rigid benchmark figures.")
    parser.add_argument("--artifact-dir", required=True, help="Primary benchmark artifact directory")
    parser.add_argument("--data-root", required=True, help="Path to ScanNet root containing scans/")
    parser.add_argument("--output-root", default=None, help="Directory for generated figures")
    parser.add_argument("--comparison-artifact-dir", default=None, help="Optional comparison benchmark artifact directory")
    parser.add_argument("--ablation-artifact-dir", default=None, help="Optional ablation benchmark artifact directory")
    parser.add_argument("--scene-index", type=int, default=0, help="Index into the saved test scene list for the hero figure")
    args = parser.parse_args()

    result = generate_scannet_paper_figures(
        ScanNetPaperFigureConfig(
            primary_artifact_dir=args.artifact_dir,
            data_root=args.data_root,
            output_root=args.output_root,
            comparison_artifact_dir=args.comparison_artifact_dir,
            ablation_artifact_dir=args.ablation_artifact_dir,
            scene_index=args.scene_index,
        )
    )
    print(f"Figure directory: {result['output_dir']}")
    print(f"Scene id: {result['scene_id']}")


if __name__ == "__main__":
    main()
