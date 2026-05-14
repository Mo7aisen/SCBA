# SCBA++: Counterfactual Consistency Auditing of Decoder-Focused CAMs for Lung Segmentation in Chest X-rays

This repository provides the implementation for the SCBA++ framework, presented at the **34th European Signal Processing Conference (EUSIPCO 2026)**.

## Overview

SCBA++ (Synthetic Counterfactual Border Audit++) is a framework for auditing the consistency of gradient-based attribution methods in medical image segmentation. By generating controlled, anatomically plausible boundary perturbations using thin-plate-spline (TPS) warping and Poisson blending, SCBA++ tests whether CAM-based explanations respond appropriately to counterfactual changes in chest X-ray lung boundaries.

Four decoder-focused CAM variants are evaluated:

- **Multilayer CAM** (proposed) -- aggregates LayerCAM from two decoder blocks
- **LayerCAM** -- single decoder layer with gradient-weighted activations
- **HiResCAM** -- element-wise activation-gradient products
- **Grad-CAM++** -- higher-order gradient weighting

Consistency is measured via three metrics: attribution mass change in the region of interest (ΔAM-ROI), center-of-attribution shift (CoA Shift), and directional consistency (DC).

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
git clone https://github.com/Mo7aisen/SCBA.git
cd SCBA
pip install -e .
```

## Datasets

The framework is evaluated on two public chest X-ray datasets:

- **JSRT** (Japanese Society of Radiological Technology): 247 posteroanterior radiographs with expert-annotated lung masks
- **Montgomery County**: 138 posteroanterior radiographs from the U.S. National Library of Medicine

Dataset preparation scripts are provided in `scba/scripts/`.

## Usage

### Running the main experiments

```bash
# Publication experiments (JSRT + Montgomery, 4 methods, 3 perturbations each)
python run_publication_experiments.py

# Ablation study (perturbation magnitude)
python run_ablation_study.py

# Sanity check (model randomization)
python run_sanity_randomization.py
```

### Using individual components

```python
from scba.cf.borders import BorderEditor, BorderEditConfig
from scba.xai.cam.seg_grad_cam import SegGradCAM

# Generate a counterfactual border perturbation
config = BorderEditConfig(radius_px=3, operation="dilate", warp_mode="tps")
editor = BorderEditor(config)
cf_image, new_mask, roi_band = editor.apply_border_edit(image, mask)

# Compute attribution map
explainer = SegGradCAM(model, device="cuda", target_layer=target_layer)
attribution = explainer(image)
```

## Project Structure

```
SCBA/
├── scba/                          # Core package
│   ├── cf/                        # Counterfactual generation (TPS warping, Poisson blending)
│   ├── data/                      # Data loaders (JSRT, Montgomery)
│   ├── metrics/                   # Consistency metrics and statistical tests
│   ├── models/                    # U-Net segmentation model
│   ├── xai/                       # Attribution methods
│   │   ├── cam/                   # Grad-CAM variants
│   │   ├── gradient/              # Integrated Gradients
│   │   └── perturb/               # Perturbation-based methods
│   └── train/                     # Training and evaluation
├── scripts/                       # Figure generation and analysis scripts
├── experiments/                   # Experiment configurations
├── run_publication_experiments.py  # Main experiment runner
└── tests/                         # Unit tests
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{mohaisen2026scbapp,
  title={{SCBA++}: Counterfactual Consistency Auditing of Decoder-Focused {CAMs} for Lung Segmentation in Chest {X}-rays},
  author={Mohaisen, Mohammed and Hull{\'a}m, G{\'a}bor},
  booktitle={Proceedings of the 34th European Signal Processing Conference (EUSIPCO)},
  year={2026}
}
```

## Acknowledgments

- Computational resources provided by [HUN-REN Cloud](https://science-cloud.hu/)
- JSRT Database: Shiraishi et al., AJR 2000
- Montgomery Dataset: Jaeger et al., QIMS 2014

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
