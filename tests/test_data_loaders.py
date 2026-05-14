"""
Unit tests for data loaders (CI-reproducible).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_png(path: Path, array_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_uint8).save(path)


def _make_jsrt_fixture(root: Path, n: int = 6) -> None:
    # Scientific justification (R3/R2): tests must not depend on external datasets
    # and must not write splits into dataset roots. We generate a tiny synthetic
    # dataset structure in a temp dir for deterministic loader validation.
    h, w = 32, 32
    for i in range(n):
        pid = f"JPCLN_{i:03d}"
        img = np.full((h, w), i * 10, dtype=np.uint8)
        left = np.zeros((h, w), dtype=np.uint8)
        right = np.zeros((h, w), dtype=np.uint8)
        left[:, : w // 2] = 255
        right[:, w // 2 :] = 255

        _write_png(root / "images" / f"{pid}.png", img)
        _write_png(root / "masks_png" / "left_lung" / f"{pid}.png", left)
        _write_png(root / "masks_png" / "right_lung" / f"{pid}.png", right)


def _make_montgomery_fixture(root: Path, n: int = 6) -> None:
    h, w = 32, 32
    for i in range(n):
        pid = f"MCUCXR_{i:04d}"
        img = np.full((h, w), 128, dtype=np.uint8)
        left = np.zeros((h, w), dtype=np.uint8)
        right = np.zeros((h, w), dtype=np.uint8)
        left[:, : w // 2] = 255
        right[:, w // 2 :] = 255

        _write_png(root / "CXR_png" / f"{pid}.png", img)
        _write_png(root / "ManualMask" / "leftMask" / f"{pid}_0.png", left)
        _write_png(root / "ManualMask" / "rightMask" / f"{pid}_0.png", right)


def test_jsrt_loader_is_self_contained_and_does_not_write_splits(tmp_path: Path) -> None:
    from scba.data.loaders.jsrt import JSRTDataset

    root = tmp_path / "jsrt"
    _make_jsrt_fixture(root)

    ds = JSRTDataset(str(root), split="train", target_size=(16, 16), return_patient_id=True)
    assert len(ds) > 0
    assert not (root / "splits.csv").exists()

    sample = ds[0]
    assert set(sample.keys()) >= {"image", "mask", "left_lung", "right_lung", "patient_id"}
    assert sample["image"].shape == (16, 16)
    assert sample["mask"].shape == (16, 16)
    assert sample["image"].dtype == np.float32
    assert sample["mask"].dtype == np.uint8


def test_montgomery_loader_is_self_contained_and_does_not_write_splits(tmp_path: Path) -> None:
    from scba.data.loaders.montgomery import MontgomeryDataset

    root = tmp_path / "montgomery"
    _make_montgomery_fixture(root)

    ds = MontgomeryDataset(str(root), split="train", target_size=(16, 16), return_patient_id=True)
    assert len(ds) > 0
    assert not (root / "splits.csv").exists()

    sample = ds[0]
    assert set(sample.keys()) >= {"image", "mask", "left_lung", "right_lung", "patient_id"}
    assert sample["image"].shape == (16, 16)
    assert sample["mask"].shape == (16, 16)
    assert sample["image"].dtype == np.float32
    assert sample["mask"].dtype == np.uint8


def test_deterministic_splits_are_repeatable_without_writing(tmp_path: Path) -> None:
    from scba.data.loaders.jsrt import JSRTDataset

    root = tmp_path / "jsrt"
    _make_jsrt_fixture(root, n=10)

    ds1 = JSRTDataset(str(root), split="train", return_patient_id=True)
    ds2 = JSRTDataset(str(root), split="train", return_patient_id=True)

    meta1 = ds1.metadata.sort_values("patient_id").reset_index(drop=True)
    meta2 = ds2.metadata.sort_values("patient_id").reset_index(drop=True)
    assert meta1["split"].tolist() == meta2["split"].tolist()
    assert not (root / "splits.csv").exists()
