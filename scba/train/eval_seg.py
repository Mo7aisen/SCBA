"""
Evaluation script for segmentation models.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scba.data.loaders.jsrt import JSRTDataset
from scba.data.loaders.montgomery import MontgomeryDataset
from scba.data.transforms.standard import ToTensor, get_composed_transform, get_val_transforms
from scba.models.unet import UNet
from scba.train.metrics import boundary_f_score, dice_coefficient, iou_score, pixel_accuracy, sensitivity_specificity


@torch.no_grad()
def evaluate_model(model, dataloader, device, compute_bf_score=True):
    """
    Evaluate model on a dataset.

    Returns:
        dict of metrics
    """
    model.eval()

    all_dice = []
    all_iou = []
    all_accuracy = []
    all_bf_scores = []
    all_sensitivity = []
    all_specificity = []

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        preds = torch.argmax(outputs, dim=1)

        # Compute metrics per sample
        batch_size = images.shape[0]
        for i in range(batch_size):
            pred = preds[i]
            target = masks[i]

            # Dice and IoU
            dice = dice_coefficient(pred.unsqueeze(0).unsqueeze(0), target.unsqueeze(0))
            iou = iou_score(pred.unsqueeze(0).unsqueeze(0), target.unsqueeze(0))
            acc = pixel_accuracy(pred.unsqueeze(0).unsqueeze(0), target.unsqueeze(0))

            all_dice.append(dice.item())
            all_iou.append(iou.item())
            all_accuracy.append(acc)

            # Sensitivity and specificity
            sens_spec = sensitivity_specificity(pred.unsqueeze(0).unsqueeze(0), target.unsqueeze(0))
            all_sensitivity.append(sens_spec["sensitivity"])
            all_specificity.append(sens_spec["specificity"])

            # Boundary F-score (slower, optional)
            if compute_bf_score:
                bf = boundary_f_score(pred.cpu().numpy(), target.cpu().numpy())
                all_bf_scores.append(bf)

    metrics = {
        "dice": np.mean(all_dice),
        "dice_std": np.std(all_dice),
        "iou": np.mean(all_iou),
        "iou_std": np.std(all_iou),
        "accuracy": np.mean(all_accuracy),
        "sensitivity": np.mean(all_sensitivity),
        "specificity": np.mean(all_specificity),
    }

    if compute_bf_score:
        metrics["bf_score"] = np.mean(all_bf_scores)
        metrics["bf_score_std"] = np.std(all_bf_scores)

    return metrics, all_dice, all_iou


def visualize_predictions(model, dataset, device, n_samples=8, save_path=None):
    """Visualize model predictions."""
    model.eval()

    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"]

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Convert image for display
        img_display = image.squeeze().cpu().numpy()
        mask_display = mask.numpy()

        # Plot
        axes[i, 0].imshow(img_display, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_display, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        # Overlay
        overlay = np.zeros((*img_display.shape, 3))
        overlay[:, :, 0] = mask_display  # GT in red
        overlay[:, :, 1] = pred  # Pred in green
        axes[i, 3].imshow(img_display, cmap="gray", alpha=0.7)
        axes[i, 3].imshow(overlay, alpha=0.3)
        axes[i, 3].set_title("Overlay (GT=R, Pred=G)")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")

    # Model
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["jsrt", "montgomery"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--data_root", type=str, help="Path to dataset root")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument("--target_size", type=int, default=1024, help="Target image size")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--n_vis", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--out", type=str, default="eval_results", help="Output directory")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)

    # Create model
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Data root
    if args.data_root is None:
        if args.data == "jsrt":
            args.data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
        elif args.data == "montgomery":
            args.data_root = "/home/mohaisen_mohammed/Datasets/Montgomery"

    # Dataset
    target_size = (args.target_size, args.target_size)
    val_transform = get_composed_transform(get_val_transforms(target_size))

    if args.data == "jsrt":
        dataset = JSRTDataset(
            args.data_root, split=args.split, transform=val_transform, return_patient_id=True
        )
    elif args.data == "montgomery":
        dataset = MontgomeryDataset(
            args.data_root, split=args.split, transform=val_transform, return_patient_id=True
        )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"Evaluating on {len(dataset)} samples from {args.data} {args.split} split")

    # Evaluate
    metrics, all_dice, all_iou = evaluate_model(model, dataloader, device)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("=" * 50)

    # Save results
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"{args.data}_{args.split}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")

    # Visualize
    if args.visualize:
        print(f"\nGenerating visualizations...")
        vis_path = out_dir / f"{args.data}_{args.split}_predictions.png"
        visualize_predictions(model, dataset, device, n_samples=args.n_vis, save_path=vis_path)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
