from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence


REQUIRED_SCAN_FILE_TYPES = (
    ".aggregation.json",
    ".sens",
    ".txt",
    "_vh_clean_2.ply",
)


def scene_id_from_index(index: int) -> str:
    return f"scene{index:04d}_00"


def build_download_manifest(*, start_scene_index: int, scene_count: int, file_types: Sequence[str] = REQUIRED_SCAN_FILE_TYPES) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    for offset in range(scene_count):
        scene_id = scene_id_from_index(start_scene_index + offset)
        for file_type in file_types:
            manifest.append({"scene_id": scene_id, "file_type": file_type})
    return manifest


def should_stop_for_target_size(*, current_size_gb: float, target_size_gb: float) -> bool:
    return current_size_gb >= target_size_gb


def current_directory_size_gb(root: Path) -> float:
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return round(total_bytes / (1024 ** 3), 2)


def run_required_file_download(
    *,
    scene_id: str,
    out_dir: Path,
    download_script: Path,
    python_executable: str,
    file_types: Sequence[str] = REQUIRED_SCAN_FILE_TYPES,
    log_path: Path | None = None,
) -> None:
    for file_type in file_types:
        command = [
            python_executable,
            str(download_script),
            "-o",
            str(out_dir),
            "--id",
            scene_id,
            "--type",
            file_type,
        ]
        completed = subprocess.run(
            command,
            input="\n\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(command)}\n")
                handle.write(completed.stdout)
                handle.write(completed.stderr)
                handle.write("\n")
        if completed.returncode != 0:
            raise RuntimeError(f"Download failed for {scene_id} {file_type}: {completed.returncode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a lean ScanNet subset with required files only.")
    parser.add_argument("--out-dir", required=True, help="ScanNet output root")
    parser.add_argument("--download-script", default="download-scannet.py", help="Path to original ScanNet downloader")
    parser.add_argument("--python-executable", default="python", help="Python interpreter used to invoke the original downloader")
    parser.add_argument("--start-scene-index", type=int, default=0, help="Starting scene index, e.g. 5 -> scene0005_00")
    parser.add_argument("--scene-count", type=int, default=1, help="Number of sequential scenes to download")
    parser.add_argument("--target-size-gb", type=float, default=None, help="Stop once output directory reaches this size")
    parser.add_argument("--log-path", default=None, help="Optional log path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path is not None else out_dir / "download_subset.log"

    for offset in range(args.scene_count):
        current_size = current_directory_size_gb(out_dir)
        if args.target_size_gb is not None and should_stop_for_target_size(
            current_size_gb=current_size,
            target_size_gb=args.target_size_gb,
        ):
            break
        scene_id = scene_id_from_index(args.start_scene_index + offset)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] current_size_gb={current_size} scene={scene_id}\n")
        run_required_file_download(
            scene_id=scene_id,
            out_dir=out_dir,
            download_script=Path(args.download_script),
            python_executable=args.python_executable,
            log_path=log_path,
        )


if __name__ == "__main__":
    main()
