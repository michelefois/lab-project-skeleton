"""Download and extract the Tiny ImageNet-200 dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_DATA_DIR = Path("tiny-imagenet")
DATASET_NAME = "tiny-imagenet-200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract Tiny ImageNet-200."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where tiny-imagenet-200 will be extracted "
        "(default: tiny-imagenet).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing dataset directory before extracting.",
    )
    return parser.parse_args()


def _is_safe_member(member: zipfile.ZipInfo, destination: Path) -> bool:
    target = (destination / member.filename).resolve()
    return target == destination or destination in target.parents


def download_dataset(data_dir: Path, *, force: bool = False) -> Path:
    data_dir = data_dir.resolve()
    dataset_dir = data_dir / DATASET_NAME
    archive_path = data_dir / f"{DATASET_NAME}.zip"

    if dataset_dir.exists():
        if not force:
            print(f"Dataset already present: {dataset_dir}")
            return dataset_dir
        shutil.rmtree(dataset_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Tiny ImageNet-200 to {archive_path}...")

    try:
        urllib.request.urlretrieve(DATASET_URL, archive_path)
    except Exception as error:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to download {DATASET_URL}: {error}") from error

    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if any(not _is_safe_member(member, data_dir) for member in archive.infolist()):
                raise RuntimeError("The downloaded archive contains an unsafe path.")
            archive.extractall(data_dir)
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Unable to extract {archive_path}: {error}") from error
    finally:
        archive_path.unlink(missing_ok=True)

    print(f"Dataset ready: {dataset_dir}")
    return dataset_dir


def main() -> int:
    args = parse_args()
    try:
        download_dataset(args.data_dir, force=args.force)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
