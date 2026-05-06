from __future__ import annotations

import argparse

from .scannet_qualitative_figure import ScanNetQualitativeFigureConfig, generate_scannet_qualitative_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paper-grade qualitative comparison figure on ScanNet scenes.")
    parser.add_argument("--data-root", required=True, help="Path to ScanNet root containing scans/")
    parser.add_argument("--came-ckpt", required=True, help="Checkpoint path for the CAME-Net model")
    parser.add_argument("--baseline-ckpt", required=True, help="Checkpoint path for the baseline model")
    parser.add_argument("--baseline-method", required=True, help="Baseline method name, e.g. pointnet or equiformer_v2_style")
    parser.add_argument("--scene-ids", nargs="+", required=True, help="One or more ScanNet scene ids to visualize")
    parser.add_argument("--output", required=True, help="Output path stem, e.g. artifacts/figures/qualitative_scannet")
    parser.add_argument("--came-method", default="came", help="Method name for the CAME checkpoint")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--render-num-points", type=int, default=12000, help="Number of points rendered per scene")
    parser.add_argument("--frame-index", type=int, default=0, help="RGB frame index used in the metadata column")
    parser.add_argument("--top-k-predictions", type=int, default=6, help="Fallback top-k labels when no score passes threshold")
    parser.add_argument("--camera-azim", type=float, default=35.0, help="Shared azimuth for all point-cloud panels")
    parser.add_argument("--camera-elev", type=float, default=20.0, help="Shared elevation for all point-cloud panels")
    parser.add_argument("--max-mask-objects", type=int, default=16, help="Maximum number of object groups evaluated for relevance")
    args = parser.parse_args()

    result = generate_scannet_qualitative_figure(
        ScanNetQualitativeFigureConfig(
            data_root=args.data_root,
            came_ckpt=args.came_ckpt,
            baseline_ckpt=args.baseline_ckpt,
            baseline_method=args.baseline_method,
            scene_ids=args.scene_ids,
            output=args.output,
            came_method=args.came_method,
            device=args.device,
            render_num_points=args.render_num_points,
            frame_index=args.frame_index,
            top_k_predictions=args.top_k_predictions,
            camera_azim=args.camera_azim,
            camera_elev=args.camera_elev,
            max_mask_objects=args.max_mask_objects,
        )
    )
    print(f"PNG: {result['png']}")
    print(f"PDF: {result['pdf']}")


if __name__ == "__main__":
    main()
