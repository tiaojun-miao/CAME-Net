"""
prepare_modelnet40.py - Download and prepare ModelNet40 dataset for CAME-Net.

This script downloads ModelNet40 dataset and organizes it into train/test splits.
Expected directory structure after running:
    ModelNet40/
    ├── airplane/
    │   ├── train/
    │   │   └── *.off
    │   └── test/
    │       └── *.off
    ├── bathtub/
    │   ├── train/
    │   └── test/
    └── ...

Usage:
    python prepare_modelnet40.py
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


MODELNET40_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "ModelNet40"


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading {url}...")
    print(f"Destination: {destination}")
    
    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print("\nDownload complete!")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def organize_modelnet40(source_dir: Path, target_dir: Path) -> None:
    """
    Organize ModelNet40 data into train/test splits.
    
    The original ModelNet40 structure has:
        ModelNet40/category/category/train/*.off
        ModelNet40/category/category/test/*.off
    
    We reorganize to:
        ModelNet40/category/train/*.off
        ModelNet40/category/test/*.off
    """
    print(f"Organizing dataset from {source_dir} to {target_dir}...")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all category directories
    for category_dir in sorted(source_dir.iterdir()):
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        print(f"Processing category: {category_name}")
        
        # Create train and test directories
        train_dir = target_dir / category_name / "train"
        test_dir = target_dir / category_name / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # The original structure has an extra subdirectory with the same name
        inner_dir = category_dir / category_name
        if not inner_dir.exists():
            print(f"  Warning: Expected subdirectory {inner_dir} not found, skipping")
            continue
        
        # Copy train files
        src_train = inner_dir / "train"
        if src_train.exists():
            for off_file in sorted(src_train.glob("*.off")):
                shutil.copy2(off_file, train_dir / off_file.name)
            print(f"  Train: {len(list(train_dir.glob('*.off')))} files")
        
        # Copy test files
        src_test = inner_dir / "test"
        if src_test.exists():
            for off_file in sorted(src_test.glob("*.off")):
                shutil.copy2(off_file, test_dir / off_file.name)
            print(f"  Test: {len(list(test_dir.glob('*.off')))} files")
    
    print("Organization complete!")


def verify_dataset(data_dir: Path) -> bool:
    """Verify the dataset structure is correct."""
    print(f"\nVerifying dataset at {data_dir}...")
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return False
    
    categories = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    print(f"Found {len(categories)} categories")
    
    total_train = 0
    total_test = 0
    
    for category_dir in categories:
        train_files = list((category_dir / "train").glob("*.off")) if (category_dir / "train").exists() else []
        test_files = list((category_dir / "test").glob("*.off")) if (category_dir / "test").exists() else []
        total_train += len(train_files)
        total_test += len(test_files)
        print(f"  {category_dir.name}: {len(train_files)} train, {len(test_files)} test")
    
    print(f"\nTotal: {total_train} training samples, {total_test} test samples")
    return total_train > 0 and total_test > 0


def prepare_modelnet40(
    data_dir: Optional[Path] = None,
    download: bool = True,
    cleanup: bool = True,
) -> Path:
    """
    Download and prepare ModelNet40 dataset.
    
    Args:
        data_dir: Directory to store the dataset (default: ./ModelNet40)
        download: Whether to download if not exists
        cleanup: Whether to remove temporary files after extraction
        
    Returns:
        Path to the prepared dataset directory
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    # Check if already prepared
    if verify_dataset(data_dir):
        print(f"\nDataset already prepared at {data_dir}")
        return data_dir
    
    if not download:
        raise FileNotFoundError(f"Dataset not found at {data_dir} and download=False")
    
    # Create temporary directory for download
    temp_dir = data_dir.parent / "temp_modelnet40"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = temp_dir / "ModelNet40.zip"
    
    try:
        # Download
        if not zip_path.exists():
            download_file(MODELNET40_URL, zip_path)
        
        # Extract
        extract_zip(zip_path, temp_dir)
        
        # Find extracted directory
        extracted_dir = temp_dir / "ModelNet40"
        if not extracted_dir.exists():
            # Sometimes it's nested
            for subdir in temp_dir.iterdir():
                if subdir.is_dir() and subdir.name != "temp_modelnet40":
                    extracted_dir = subdir / "ModelNet40"
                    if extracted_dir.exists():
                        break
        
        if not extracted_dir.exists():
            raise FileNotFoundError(f"Could not find extracted ModelNet40 directory")
        
        # Organize
        organize_modelnet40(extracted_dir, data_dir)
        
        # Verify
        if not verify_dataset(data_dir):
            raise RuntimeError("Dataset verification failed after preparation")
        
        print(f"\nDataset successfully prepared at {data_dir}")
        return data_dir
        
    finally:
        # Cleanup
        if cleanup and temp_dir.exists():
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare ModelNet40 dataset for CAME-Net")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to store the dataset (default: ./ModelNet40)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download if dataset not found",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary files after extraction",
    )
    
    args = parser.parse_args()
    
    prepare_modelnet40(
        data_dir=args.data_dir,
        download=not args.no_download,
        cleanup=not args.no_cleanup,
    )
