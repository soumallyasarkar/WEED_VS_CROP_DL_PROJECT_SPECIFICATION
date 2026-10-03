#!/usr/bin/env python3
"""Dataset download and setup script for CropAndWeed.

Downloads annotations and image archives from the official CropAndWeed endpoint,
extracts them to data/raw/cropandweed, and maps annotations to CropOrWeed2.
"""

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

URL_PREFIX = "https://vitro-testing.com/wp-content/uploads/2022/12/"

ARCHIVES = {
    "annotations": "cropandweed_annotations.tar",
    "images1": "cropandweed_images1of4.tar",
    "images2": "cropandweed_images2of4.tar",
    "images3": "cropandweed_images3of4.tar",
    "images4": "cropandweed_images4of4.tar",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {url} to {output_path}...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=temp_path, reporthook=t.update_to)
    temp_path.rename(output_path)


def extract_tar(tar_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {tar_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(extract_dir)


def map_croporweed2(data_root: Path) -> None:
    """Run CropOrWeed2 mapping using the official script or utility."""
    print("Mapping annotations to CropOrWeed2 variant...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "cropandweed-dataset" / "cnw"))
    try:
        from map_dataset import map_dataset  # type: ignore
        bboxes_dir = data_root / "bboxes"
        labelids_dir = data_root / "labelIds"
        
        # Verify structure: raw annotations are often extracted to bboxes/CropAndWeed or directly bboxes/
        source_bboxes = bboxes_dir / "CropAndWeed"
        if not source_bboxes.exists() and bboxes_dir.exists():
            # If bboxes files are directly in bboxes/, organize into bboxes/CropAndWeed
            csv_files = [f for f in bboxes_dir.iterdir() if f.is_file() and f.suffix == ".csv"]
            if csv_files:
                temp_dir = data_root / "CropAndWeed_temp"
                temp_dir.mkdir(exist_ok=True)
                for f in csv_files:
                    shutil.move(str(f), str(temp_dir / f.name))
                source_bboxes.mkdir(parents=True, exist_ok=True)
                for f in temp_dir.iterdir():
                    shutil.move(str(f), str(source_bboxes / f.name))
                temp_dir.rmdir()

        map_dataset(str(bboxes_dir), str(labelids_dir), "CropAndWeed", "CropOrWeed2")
        print("CropOrWeed2 mapping completed successfully.")
    except Exception as e:
        print(f"Error during mapping: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare CropAndWeed dataset.")
    parser.add_argument("--data_dir", type=str, default="data/raw/cropandweed", help="Target directory")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["annotations", "images1"],
        choices=list(ARCHIVES.keys()) + ["all"],
        help="Archives to download (annotations, images1, images2, images3, images4, or all)",
    )
    parser.add_argument("--skip_download", action="store_true", help="Skip download if archives already exist")
    parser.add_argument("--skip_mapping", action="store_true", help="Skip annotation mapping")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    archives_dir = data_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    selected_parts = list(ARCHIVES.keys()) if "all" in args.parts else args.parts

    # Always ensure annotations is included if not skipping download
    if not args.skip_download and "annotations" not in selected_parts:
        selected_parts = ["annotations"] + selected_parts

    for part in selected_parts:
        archive_name = ARCHIVES[part]
        archive_url = f"{URL_PREFIX}{archive_name}"
        archive_path = archives_dir / archive_name

        if not archive_path.exists() and not args.skip_download:
            download_file(archive_url, archive_path)

        if archive_path.exists():
            extract_tar(archive_path, data_dir)

    # Copy any sample images from the third_party repo if present
    repo_images = Path(__file__).resolve().parent.parent / "third_party" / "cropandweed-dataset" / "images"
    target_images = data_dir / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    if repo_images.exists():
        for img in repo_images.iterdir():
            if img.is_file() and not (target_images / img.name).exists():
                shutil.copy(str(img), str(target_images / img.name))

    if not args.skip_mapping:
        map_croporweed2(data_dir)

    print("=" * 60)
    print("Dataset setup finished!")
    print(f"Location: {data_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
