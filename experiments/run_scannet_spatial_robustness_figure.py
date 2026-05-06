from __future__ import annotations

import argparse

from .scannet_spatial_robustness_figure import (
    ScanNetSpatialRobustnessFigureConfig,
    generate_scannet_spatial_robustness_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a ScanNet spatial robustness qualitative figure.")
    parser.add_argument("--data-root", required=True, help="Path to ScanNet root containing scans/")
    parser.add_argument("--came-ckpt", required=True, help="Checkpoint path for the CAME-Net model")
    parser.add_argument("--baseline-ckpt", required=True, help="Checkpoint path for the baseline model")
    parser.add_argument("--baseline-method", required=True, help="Baseline method name, e.g. pointnet")
    parser.add_argument("--scene-ids", nargs="+", required=True, help="ScanNet scene ids, one per row")
    parser.add_argument("--transform-variants", nargs="+", required=True, help="Transform variants aligned with scene ids, e.g. rot_z_30 tx_0p2")
    parser.add_argument("--output", required=True, help="Output path stem, e.g. artifacts/figures/robustness_scannet")
    parser.add_argument("--came-method", default="came", help="Method name for the CAME checkpoint")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--render-num-points", type=int, default=12000, help="Number of rendered points per scene")
    parser.add_argument("--top-k-predictions", type=int, default=6, help="Fallback top-k labels when no score passes threshold")
    parser.add_argument("--camera-azim", type=float, default=35.0, help="Shared azimuth for all panels")
    parser.add_argument("--camera-elev", type=float, default=20.0, help="Shared elevation for all panels")
    args = parser.parse_args()

    result = generate_scannet_spatial_robustness_figure(
        ScanNetSpatialRobustnessFigureConfig(
            data_root=args.data_root,
            came_ckpt=args.came_ckpt,
            baseline_ckpt=args.baseline_ckpt,
            baseline_method=args.baseline_method,
            scene_ids=args.scene_ids,
            transform_variants=args.transform_variants,
            output=args.output,
            came_method=args.came_method,
            device=args.device,
            render_num_points=args.render_num_points,
            top_k_predictions=args.top_k_predictions,
            camera_azim=args.camera_azim,
            camera_elev=args.camera_elev,
        )
    )
    print(f"PNG: {result['png']}")
    print(f"PDF: {result['pdf']}")


if __name__ == "__main__":
    main()
