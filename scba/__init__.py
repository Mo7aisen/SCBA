"""
SCBA: Synthetic Counterfactual Border Audit

A systematic framework for auditing segmentation explainability in chest X-rays.
"""

__version__ = "0.1.0"
__author__ = "AI for Medical Imaging Lab"

from scba import cf, data, metrics, models, robustness, train, ui, xai

__all__ = ["data", "models", "xai", "cf", "metrics", "robustness", "ui", "train"]
