"""
Smoke test for the publication pipeline with a synthetic JSRT fixture.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _write_png(path: Path, array_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_uint8).save(path)


def _make_jsrt_fixture(root: Path, n: int = 10) -> None:
    h, w = 32, 32
    for i in range(n):
        pid = f"JPCLN_{i:03d}"
        img = np.full((h, w), 128, dtype=np.uint8)
        left = np.zeros((h, w), dtype=np.uint8)
        right = np.zeros((h, w), dtype=np.uint8)
        margin = 2
        left[margin : h - margin, margin : (w // 2) - margin] = 255
        right[margin : h - margin, (w // 2) + margin : w - margin] = 255

        _write_png(root / "images" / f"{pid}.png", img)
        _write_png(root / "masks_png" / "left_lung" / f"{pid}.png", left)
        _write_png(root / "masks_png" / "right_lung" / f"{pid}.png", right)


def _write_dummy_checkpoint(path: Path) -> None:
    from scba.models.unet import UNet

    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    with torch.no_grad():
        model.outc.conv.weight.zero_()
        model.outc.conv.bias.zero_()
        model.outc.conv.bias[1] = 10.0
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 0, "best_score": 0.0},
        path,
    )


def test_publication_pipeline_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "jsrt"
    _make_jsrt_fixture(data_root)

    model_path = tmp_path / "dummy_model.pt"
    _write_dummy_checkpoint(model_path)

    output_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env["SCBA_QUICK"] = "1"
    env["SCBA_DISABLE_TQDM"] = "1"

    cmd = [
        sys.executable,
        "run_publication_experiments.py",
        "--dataset",
        "jsrt",
        "--model-path",
        str(model_path),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--limit-samples",
        "1",
        "--target-size",
        "32",
        "--auc-steps",
        "2",
        "--mask-source",
        "gt",
    ]

    subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    results_file = output_dir / "scba_publication_results.json"
    assert results_file.exists()
