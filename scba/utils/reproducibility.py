from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed Python/NumPy/PyTorch RNGs for reproducible scientific experiments.

    Scientific justification: SCBA metrics (and some XAI baselines) involve
    randomized sampling (e.g., RISE masks, LIME/SHAP sampling, bootstrap CIs).
    If seeds are not fixed consistently, reported p-values/effect sizes and
    figures can change across runs, which invalidates a manuscript-level claim
    of reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    if deterministic:
        # Scientific justification: deterministic kernels reduce run-to-run
        # variation that can otherwise masquerade as a method effect.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

