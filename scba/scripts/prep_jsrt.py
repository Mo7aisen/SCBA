"""
JSRT dataset preparation script.

Verifies dataset structure, creates splits, and generates summary statistics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def verify_jsrt_structure(data_root: Path) -> bool:
    """Verify JSRT dataset has expected structure."""
    required_dirs = [
        data_root / "images",
        data_root / "masks_png" / "left_lung",
        data_root / "masks_png" / "right_lung",
    ]

    for d in required_dirs:
        if not d.exists():
            print(f"❌ Missing directory: {d}")
            return False

    print("✓ Directory structure verified")
    return True


def analyze_dataset(data_root: Path, out_dir: Path):
    """Analyze JSRT dataset and generate statistics."""
    images_dir = data_root / "images"
    masks_dir = data_root / "masks_png"

    image_files = sorted(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

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

        # Load masks
        left_mask_path = masks_dir / "left_lung" / f"{patient_id}.png"
        right_mask_path = masks_dir / "right_lung" / f"{patient_id}.png"

        if not (left_mask_path.exists() and right_mask_path.exists()):
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
    df.to_csv(out_dir / "jsrt_statistics.csv", index=False)

    # Print summary
    print("\n" + "=" * 50)
    print("JSRT Dataset Summary")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Image shapes: {df['image_shape'].value_counts().to_dict()}")
    print(f"Mean lung area: {df['mask_area'].mean():.0f} pixels")
    print(f"Mean left lung: {df['left_lung_area'].mean():.0f} pixels")
    print(f"Mean right lung: {df['right_lung_area'].mean():.0f} pixels")
    print("=" * 50)

    return df


def visualize_samples(data_root: Path, out_dir: Path, n_samples=5):
    """Visualize random samples from the dataset."""
    from scba.data.loaders.jsrt import JSRTDataset

    dataset = JSRTDataset(data_root, split="train", target_size=(512, 512), return_patient_id=True)

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
    plt.savefig(out_dir / "jsrt_samples.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to {out_dir / 'jsrt_samples.png'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare JSRT dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/mohaisen_mohammed/Datasets/JSRT",
        help="Path to JSRT dataset root",
    )
    parser.add_argument(
        "--out", type=str, default="data/jsrt", help="Output directory for metadata"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization of samples"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Preparing JSRT dataset...")

    # Verify structure
    if not verify_jsrt_structure(data_root):
        print("❌ Dataset structure verification failed")
        return

    # Analyze dataset
    stats_df = analyze_dataset(data_root, out_dir)

    # Create splits
    print("\n📊 Creating data splits...")
    from scba.data.loaders.jsrt import JSRTDataset

    dataset = JSRTDataset(data_root, split="train")
    split_counts = dataset.get_split_counts()
    print(f"Split counts: {split_counts}")

    # Visualize if requested
    if args.visualize:
        print("\n📸 Generating visualizations...")
        visualize_samples(data_root, out_dir)

    print("\n✅ JSRT dataset preparation complete!")


if __name__ == "__main__":
    main()
