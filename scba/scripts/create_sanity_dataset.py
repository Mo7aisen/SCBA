"""
Create a small sanity dataset for CI and demo purposes.

Extracts a few samples from the full datasets for quick testing.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


def create_jsrt_sanity(source_root: Path, target_root: Path, n_samples=5):
    """
    Create sanity subset of JSRT dataset.

    Args:
        source_root: Path to full JSRT dataset
        target_root: Path to save sanity dataset
        n_samples: Number of samples to extract
    """
    target_root.mkdir(parents=True, exist_ok=True)

    # Create directories
    (target_root / "images").mkdir(exist_ok=True)
    (target_root / "masks_png" / "left_lung").mkdir(parents=True, exist_ok=True)
    (target_root / "masks_png" / "right_lung").mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = sorted((source_root / "images").glob("*.png"))

    # Sample deterministically
    np.random.seed(42)
    indices = np.random.choice(len(image_files), size=min(n_samples, len(image_files)), replace=False)
    sampled_files = [image_files[i] for i in indices]

    print(f"Creating JSRT sanity dataset with {len(sampled_files)} samples...")

    for img_path in sampled_files:
        patient_id = img_path.stem

        # Copy image
        shutil.copy(img_path, target_root / "images" / img_path.name)

        # Copy masks
        left_src = source_root / "masks_png" / "left_lung" / f"{patient_id}.png"
        right_src = source_root / "masks_png" / "right_lung" / f"{patient_id}.png"

        if left_src.exists():
            shutil.copy(left_src, target_root / "masks_png" / "left_lung" / f"{patient_id}.png")
        if right_src.exists():
            shutil.copy(right_src, target_root / "masks_png" / "right_lung" / f"{patient_id}.png")

    print(f"✓ JSRT sanity dataset created at {target_root}")


def create_montgomery_sanity(source_root: Path, target_root: Path, n_samples=5):
    """
    Create sanity subset of Montgomery dataset.

    Args:
        source_root: Path to full Montgomery dataset
        target_root: Path to save sanity dataset
        n_samples: Number of samples to extract
    """
    target_root.mkdir(parents=True, exist_ok=True)

    # Create directories
    (target_root / "CXR_png").mkdir(exist_ok=True)
    (target_root / "ManualMask" / "leftMask").mkdir(parents=True, exist_ok=True)
    (target_root / "ManualMask" / "rightMask").mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = sorted((source_root / "CXR_png").glob("MCUCXR_*.png"))

    # Sample deterministically
    np.random.seed(42)
    indices = np.random.choice(len(image_files), size=min(n_samples, len(image_files)), replace=False)
    sampled_files = [image_files[i] for i in indices]

    print(f"Creating Montgomery sanity dataset with {len(sampled_files)} samples...")

    def find_mask(patient_id, side):
        """Find mask with either _0 or _1 suffix."""
        mask_dir = source_root / "ManualMask" / f"{side}Mask"
        for suffix in ["_0", "_1"]:
            mask_path = mask_dir / f"{patient_id}{suffix}.png"
            if mask_path.exists():
                return mask_path
        return None

    for img_path in sampled_files:
        patient_id = img_path.stem

        # Copy image
        shutil.copy(img_path, target_root / "CXR_png" / img_path.name)

        # Copy masks
        left_src = find_mask(patient_id, "left")
        right_src = find_mask(patient_id, "right")

        if left_src:
            shutil.copy(left_src, target_root / "ManualMask" / "leftMask" / left_src.name)
        if right_src:
            shutil.copy(right_src, target_root / "ManualMask" / "rightMask" / right_src.name)

    print(f"✓ Montgomery sanity dataset created at {target_root}")


def main():
    parser = argparse.ArgumentParser(description="Create sanity datasets for CI and demo")
    parser.add_argument(
        "--jsrt_source",
        type=str,
        default="/home/mohaisen_mohammed/Datasets/JSRT",
        help="Path to full JSRT dataset",
    )
    parser.add_argument(
        "--montgomery_source",
        type=str,
        default="/home/mohaisen_mohammed/Datasets/Montgomery",
        help="Path to full Montgomery dataset",
    )
    parser.add_argument(
        "--output", type=str, default="assets/sanity_data", help="Output directory for sanity datasets"
    )
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples per dataset")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create JSRT sanity dataset
    jsrt_source = Path(args.jsrt_source)
    if jsrt_source.exists():
        create_jsrt_sanity(jsrt_source, output_dir / "jsrt", args.n_samples)
    else:
        print(f"⚠ JSRT source not found: {jsrt_source}")

    # Create Montgomery sanity dataset
    montgomery_source = Path(args.montgomery_source)
    if montgomery_source.exists():
        create_montgomery_sanity(montgomery_source, output_dir / "montgomery", args.n_samples)
    else:
        print(f"⚠ Montgomery source not found: {montgomery_source}")

    print("\n✅ Sanity datasets created successfully!")
    print(f"Location: {output_dir}")


if __name__ == "__main__":
    main()
