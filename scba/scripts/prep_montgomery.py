"""
Montgomery dataset preparation script.

Verifies dataset structure, creates splits, and generates summary statistics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def verify_montgomery_structure(data_root: Path) -> bool:
    """Verify Montgomery dataset has expected structure."""
    required_dirs = [
        data_root / "CXR_png",
        data_root / "ManualMask" / "leftMask",
        data_root / "ManualMask" / "rightMask",
    ]

    for d in required_dirs:
        if not d.exists():
            print(f"❌ Missing directory: {d}")
            return False

    print("✓ Directory structure verified")
    return True


def analyze_dataset(data_root: Path, out_dir: Path):
    """Analyze Montgomery dataset and generate statistics."""
    images_dir = data_root / "CXR_png"
    masks_dir = data_root / "ManualMask"

    image_files = sorted(images_dir.glob("MCUCXR_*.png"))
    print(f"Found {len(image_files)} images")

    def find_mask(patient_id, side):
        """Find mask with either _0 or _1 suffix."""
        mask_dir = masks_dir / f"{side}Mask"
        for suffix in ["_0", "_1"]:
            mask_path = mask_dir / f"{patient_id}{suffix}.png"
            if mask_path.exists():
                return mask_path
        return None

    stats = {
        "patient_id": [],
        "image_shape": [],
        "image_min": [],
        "image_max": [],
        "image_mean": [],
        "mask_area": [],
        "left_lung_area": [],
        "right_lung_area": [],
    }

    for img_path in tqdm(image_files, desc="Analyzing images"):
        patient_id = img_path.stem

        # Load image
        img = np.array(Image.open(img_path).convert("L"))

        # Find masks
        left_mask_path = find_mask(patient_id, "left")
        right_mask_path = find_mask(patient_id, "right")

        if not (left_mask_path and right_mask_path):
            print(f"⚠ Missing masks for {patient_id}")
            continue

        left_mask = np.array(Image.open(left_mask_path).convert("L"))
        right_mask = np.array(Image.open(right_mask_path).convert("L"))

        # Binarize
        left_mask = (left_mask > 127).astype(np.uint8)
        right_mask = (right_mask > 127).astype(np.uint8)
        combined_mask = np.clip(left_mask + right_mask, 0, 1)

        # Collect stats
        stats["patient_id"].append(patient_id)
        stats["image_shape"].append(img.shape)
        stats["image_min"].append(img.min())
        stats["image_max"].append(img.max())
        stats["image_mean"].append(img.mean())
        stats["mask_area"].append(combined_mask.sum())
        stats["left_lung_area"].append(left_mask.sum())
        stats["right_lung_area"].append(right_mask.sum())

    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "montgomery_statistics.csv", index=False)

    # Print summary
    print("\n" + "=" * 50)
    print("Montgomery Dataset Summary")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Unique image shapes: {df['image_shape'].nunique()}")
    print(f"Mean lung area: {df['mask_area'].mean():.0f} pixels")
    print(f"Mean left lung: {df['left_lung_area'].mean():.0f} pixels")
    print(f"Mean right lung: {df['right_lung_area'].mean():.0f} pixels")
    print("=" * 50)

    return df


def visualize_samples(data_root: Path, out_dir: Path, n_samples=5):
    """Visualize random samples from the dataset."""
    from scba.data.loaders.montgomery import MontgomeryDataset

    dataset = MontgomeryDataset(
        data_root, split="train", target_size=(512, 512), return_patient_id=True
    )

    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]

        image = sample["image"]
        mask = sample["mask"]
        left = sample["left_lung"]
        right = sample["right_lung"]
        patient_id = sample["patient_id"]

        # Plot
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"{patient_id}\nOriginal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Combined Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(left, cmap="Blues")
        axes[i, 2].set_title("Left Lung")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(right, cmap="Reds")
        axes[i, 3].set_title("Right Lung")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "montgomery_samples.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to {out_dir / 'montgomery_samples.png'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Montgomery dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/mohaisen_mohammed/Datasets/Montgomery",
        help="Path to Montgomery dataset root",
    )
    parser.add_argument(
        "--out", type=str, default="data/montgomery", help="Output directory for metadata"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization of samples"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Preparing Montgomery dataset...")

    # Verify structure
    if not verify_montgomery_structure(data_root):
        print("❌ Dataset structure verification failed")
        return

    # Analyze dataset
    stats_df = analyze_dataset(data_root, out_dir)

    # Create splits
    print("\n📊 Creating data splits...")
    from scba.data.loaders.montgomery import MontgomeryDataset

    dataset = MontgomeryDataset(data_root, split="train")
    split_counts = dataset.get_split_counts()
    print(f"Split counts: {split_counts}")

    # Visualize if requested
    if args.visualize:
        print("\n📸 Generating visualizations...")
        visualize_samples(data_root, out_dir)

    print("\n✅ Montgomery dataset preparation complete!")


if __name__ == "__main__":
    main()
