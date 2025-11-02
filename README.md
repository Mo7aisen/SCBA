# Synthetic Counterfactual Border Audit (SCBA)

**A systematic framework for auditing segmentation explainability in chest X-rays using synthetic counterfactuals**

## Overview

SCBA introduces a novel framework for evaluating the consistency of gradient-based attribution methods (Grad-CAM variants) in medical image segmentation. By generating controlled boundary perturbations in chest X-ray lung segmentation, we systematically test whether explanation methods respond appropriately to counterfactual changes.

## Key Features

- **Synthetic Counterfactual Generation**: Morphological border perturbations with Poisson blending
- **Grad-CAM Variants**: Seg-Grad-CAM, HiResCAM, Grad-CAM++ implementations
- **Novel Consistency Metrics**: Attribution Mass in ROI (ΔAM-ROI), Center of Attribution Shift, Directional Consistency
- **Statistical Validation**: Bootstrap confidence intervals, hypothesis testing with Bonferroni correction
- **Conference Manuscript**: Complete LaTeX source with figures and references

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/Mo7aisen/SCBA.git
cd SCBA

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Experiments

```bash
# Run main publication experiments
python run_publication_experiments.py

# Run ablation study (perturbation magnitude)
python run_ablation_study.py

# Generate visual comparison figures
python generate_visual_comparison.py
```

### Using Individual Components

```python
from scba.cf.borders import apply_border_edit
from scba.xai.cam.seg_grad_cam import SegGradCAM

# Generate counterfactual
img_cf, mask_cf, roi = apply_border_edit(
    image=image,
    mask=predicted_mask,
    radius_px=3,
    operation='dilate'
)

# Compute attribution
explainer = SegGradCAM(model, device='cuda', target_layer=model.outc.conv)
attribution = explainer(image)
```

## Project Structure

```
SCBA/
├── scba/                       # Core package
│   ├── cf/                     # Counterfactual generation
│   │   └── borders.py          # Border manipulation
│   ├── data/                   # Data loaders
│   │   └── loaders/
│   │       └── jsrt.py         # JSRT dataset
│   ├── metrics/                # Evaluation metrics
│   │   └── cf_consistency.py   # Consistency metrics
│   ├── models/                 # Segmentation models
│   │   └── unet.py             # U-Net architecture
│   └── xai/                    # XAI methods
│       └── cam/                # Grad-CAM variants
│           ├── seg_grad_cam.py
│           ├── hires_cam.py
│           └── grad_cam_pp.py
├── manuscript/                 # Conference paper
│   ├── scba_manuscript.tex
│   ├── references.bib
│   └── figures/
├── run_*.py                    # Experiment scripts
└── tests/                      # Unit tests
```

## Datasets

The framework is evaluated on the **JSRT** (Japanese Society of Radiological Technology) chest X-ray dataset:

- 247 posteroanterior chest radiographs
- Expert-annotated lung segmentation masks
- Images: 2048×2048 resolution

## Methods

### XAI Methods

**Grad-CAM Variants:**
- Seg-Grad-CAM
- HiResCAM
- Grad-CAM++

### Counterfactual Generation

**Border Perturbations:**
- Morphological dilation/erosion (radius 2-3 pixels)
- Poisson blending for realistic synthesis

### Consistency Metrics

- **ΔAM-ROI**: Attribution mass change in region of interest
- **CoA Shift**: Center of attribution spatial displacement
- **Directional Consistency**: Attribution movement toward/from ROI

## Results

Key findings from evaluation on 38 JSRT test samples:

- **Method Equivalence**: Seg-Grad-CAM, HiResCAM produce nearly identical outputs (Dice > 0.99)
- **Grad-CAM++ Distinctiveness**: Shows significantly different behavior with negative ΔAM-ROI (-0.0011 vs +0.0039, p<0.001)
- **Stability**: Grad-CAM++ exhibits 30% smaller spatial shifts (3.42 vs 4.92 pixels)
- **Statistical Significance**: Large effect sizes (Cohen's d = 0.92) with rigorous validation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mohammed2025scba,
  title={Counterfactual Consistency Auditing of Grad-CAM Methods for Lung Segmentation in Chest X-rays},
  author={Mohammed, Mohaisen},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- JSRT Database: Japanese Society of Radiological Technology
- Montgomery & Shenzhen datasets: Jaeger et al., 2014
- Seg-Grad-CAM: Vinogradova et al., AAAI 2020

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.
