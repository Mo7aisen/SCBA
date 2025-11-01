"""
Unit tests for data loaders.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.mark.skipif(
    not Path("/home/mohaisen_mohammed/Datasets/JSRT").exists(),
    reason="JSRT dataset not available"
)
def test_jsrt_loader():
    """Test JSRT dataset loader."""
    from scba.data.loaders.jsrt import JSRTDataset

    dataset = JSRTDataset(
        "/home/mohaisen_mohammed/Datasets/JSRT",
        split="train",
        target_size=(512, 512),
    )

    assert len(dataset) > 0, "Dataset should not be empty"

    # Test sample loading
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    assert "left_lung" in sample
    assert "right_lung" in sample

    # Check shapes
    assert sample["image"].shape == (512, 512)
    assert sample["mask"].shape == (512, 512)

    # Check dtypes
    assert sample["image"].dtype == np.float32
    assert sample["mask"].dtype == np.uint8

    # Check value ranges
    assert 0 <= sample["image"].min() <= 1
    assert 0 <= sample["image"].max() <= 1
    assert sample["mask"].max() <= 1


@pytest.mark.skipif(
    not Path("/home/mohaisen_mohammed/Datasets/Montgomery").exists(),
    reason="Montgomery dataset not available"
)
def test_montgomery_loader():
    """Test Montgomery dataset loader."""
    from scba.data.loaders.montgomery import MontgomeryDataset

    dataset = MontgomeryDataset(
        "/home/mohaisen_mohammed/Datasets/Montgomery",
        split="train",
        target_size=(512, 512),
    )

    assert len(dataset) > 0, "Dataset should not be empty"

    # Test sample loading
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    assert "left_lung" in sample
    assert "right_lung" in sample

    # Check shapes
    assert sample["image"].shape == (512, 512)
    assert sample["mask"].shape == (512, 512)

    # Check dtypes
    assert sample["image"].dtype == np.float32
    assert sample["mask"].dtype == np.uint8

    # Check value ranges
    assert 0 <= sample["image"].min() <= 1
    assert 0 <= sample["image"].max() <= 1
    assert sample["mask"].max() <= 1


def test_deterministic_splits():
    """Test that splits are deterministic."""
    from scba.data.loaders.jsrt import JSRTDataset
    import tempfile
    import shutil

    # This test would require setting up a temporary dataset
    # For now, we'll skip the implementation
    pytest.skip("Requires temporary dataset setup")
