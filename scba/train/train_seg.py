"""
Training script for segmentation models.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scba.data.loaders.jsrt import get_jsrt_dataloaders
from scba.data.loaders.montgomery import get_montgomery_dataloaders
from scba.data.transforms.standard import (
    ToTensor,
    get_composed_transform,
    get_train_transforms,
    get_val_transforms,
)
from scba.models.unet import get_unet
from scba.train.losses import get_loss
from scba.train.metrics import MetricTracker, dice_coefficient, iou_score


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=True):
    """Train for one epoch."""
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # Compute metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)

        tracker.update(loss=loss.item(), dice=dice.item(), iou=iou.item())
        pbar.set_postfix(loss=loss.item(), dice=dice.item())

    return tracker.compute()


@torch.no_grad()
def val_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        dice = dice_coefficient(outputs, masks)
        iou = iou_score(outputs, masks)

        tracker.update(loss=loss.item(), dice=dice.item(), iou=iou.item())
        pbar.set_postfix(loss=loss.item(), dice=dice.item())

    return tracker.compute()


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["jsrt", "montgomery"],
        help="Dataset to use",
    )
    parser.add_argument("--data_root", type=str, help="Path to dataset root (optional)")
    parser.add_argument("--target_size", type=int, default=1024, help="Target image size")

    # Model
    parser.add_argument("--arch", type=str, default="unet", choices=["unet"], help="Model architecture")
    parser.add_argument("--base_features", type=int, default=64, help="Base features for U-Net")

    # Training
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--loss", type=str, default="dice_bce", help="Loss function")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")

    # Optimization
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler")
    parser.add_argument("--early_stop", type=int, default=15, help="Early stopping patience")

    # Output
    parser.add_argument("--save", type=str, required=True, help="Path to save model")
    parser.add_argument("--save_dir", type=str, default="runs", help="Directory for checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data root
    if args.data_root is None:
        if args.data == "jsrt":
            args.data_root = "/home/mohaisen_mohammed/Datasets/JSRT"
        elif args.data == "montgomery":
            args.data_root = "/home/mohaisen_mohammed/Datasets/Montgomery"

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    target_size = (args.target_size, args.target_size)
    train_transform = get_composed_transform(get_train_transforms(target_size))
    val_transform = get_composed_transform(get_val_transforms(target_size))

    # Dataloaders
    print(f"Loading {args.data} dataset...")
    if args.data == "jsrt":
        dataloaders = get_jsrt_dataloaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=target_size,
            train_transform=train_transform,
            val_transform=val_transform,
        )
    elif args.data == "montgomery":
        dataloaders = get_montgomery_dataloaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=target_size,
            train_transform=train_transform,
            val_transform=val_transform,
        )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    print(f"Creating {args.arch} model...")
    model = get_unet(n_channels=1, n_classes=2, base_features=args.base_features)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss
    criterion = get_loss(args.loss)

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
        )

    # Scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Mixed precision scaler
    scaler = GradScaler() if args.amp else None

    # Training loop
    best_dice = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args.amp)
        print(f"Train - {' | '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])}")

        # Validate
        val_metrics = val_epoch(model, val_loader, criterion, device)
        print(f"Val   - {' | '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")

        # Scheduler step
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["dice"])
            else:
                scheduler.step()

        # Save history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }
            torch.save(checkpoint, args.save)
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"\nEarly stopping triggered (patience: {args.early_stop})")
            break

    # Save training history
    history_path = save_dir / f"{Path(args.save).stem}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {args.save}")
    print("=" * 50)


if __name__ == "__main__":
    main()
