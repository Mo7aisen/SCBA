# Synthetic Counterfactual Border Audit (SCBA)

**A systematic framework for auditing segmentation explainability in chest X-rays using synthetic counterfactuals**

## Overview

SCBA introduces a novel methodology to evaluate whether explainable AI (XAI) methods for medical image segmentation produce faithful, stable, and clinically meaningful attributions. By generating controlled, realistic perturbations at organ borders and inserting synthetic lesions, we can systematically audit whether explanations track causal changes in the image.

## Key Features

- **Synthetic Counterfactual Generation**: Realistic border edits and Gaussian nodule surrogates
- **Comprehensive XAI Baselines**: 12+ methods including Seg-Grad-CAM, RISE, LIME, SHAP, IG, LRP
- **Novel Metrics**: Attribution Mass in ROI (AM-ROI), Center of Attribution shift (CoA-Δ), Directional Consistency
- **Robustness Suite**: Benign perturbations, domain shift, adversarial tests, sanity checks
- **Clinical Deployment**: DICOM SEG/SR export, PACS integration, triptych viewer
- **User Study Framework**: Application-grounded evaluation with radiologists

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (recommended for GPU acceleration)
- 16GB+ RAM
- 8GB+ GPU memory (for training)

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/scba.git
cd scba

# Create conda environment
conda env create -f env/environment.yml
conda activate scba

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### 1. Prepare Datasets

```bash
# JSRT dataset
python -m scba.scripts.prep_jsrt --data_root /path/to/jsrt --out data/jsrt

# Montgomery dataset
python -m scba.scripts.prep_montgomery --data_root /path/to/montgomery --out data/montgomery
```

### 2. Train Baseline Model

```bash
python -m scba.train.train_seg \
  --data jsrt \
  --arch unet \
  --epochs 80 \
  --amp \
  --save runs/jsrt_unet.pt
```

### 3. Run Counterfactual Sweep

```bash
python -m scba.scripts.run_cf_sweep \
  --data jsrt \
  --ckpt runs/jsrt_unet.pt \
  --methods seg_grad_cam,rise,lime,shap \
  --border_radii 2,4,8 \
  --lesion_sigmas 4,8,12 \
  --out results/jsrt_cf
```

### 4. Launch Interactive UI

```bash
python -m scba.ui.app --demo assets/demo_jsrt
```

## Project Structure

```
scba/
├── scba/
│   ├── data/           # Data loaders and transforms
│   ├── models/         # Segmentation architectures
│   ├── xai/           # XAI methods (CAM, perturbation-based)
│   ├── cf/            # Counterfactual generators
│   ├── metrics/       # Evaluation metrics
│   ├── robustness/    # Robustness test suite
│   ├── ui/            # Interactive viewer and study mode
│   ├── train/         # Training and evaluation scripts
│   └── scripts/       # Data prep and sweep scripts
├── experiments/       # Configs and results
├── tests/            # Unit and integration tests
└── env/              # Environment specifications
```

## Datasets

SCBA is designed for chest X-ray datasets with lung segmentation masks:

- **JSRT**: 247 PA chest X-rays (2048×2048)
- **Montgomery**: 138 PA chest X-rays with TB screening
- **Optional**: Shenzhen, SIIM-ACR Pneumothorax, CheXmask

See `docs/datasets.md` for detailed preparation instructions.

## Methods

### XAI Methods Implemented

**Gradient-based:**
- Seg-Grad-CAM, Seg-XRes-CAM, HiResCAM
- Grad-CAM++, Guided Grad-CAM
- Integrated Gradients, LRP

**Perturbation-based:**
- RISE, LIME, SHAP, Occlusion

### Counterfactual Perturbations

1. **Border Edits**: Morphological operations + TPS warping + Poisson blending
2. **Gaussian Nodules**: Realistic lesion surrogates with intensity matching
3. **Repair Operations**: Inpainting to reverse perturbations

### Evaluation Metrics

- **Faithfulness**: Deletion/Insertion AUC
- **CF Consistency**: AM-ROI, ΔAM-ROI, CoA-Δ, Directional Consistency
- **Stability**: SSIM, Pearson correlation under benign noise
- **Localization**: IoU, pointing game accuracy
- **Sanity Checks**: Parameter/input randomization

## Citation

If you use SCBA in your research, please cite:

```bibtex
@inproceedings{scba2024,
  title={Synthetic Counterfactual Border Audit for Segmentation Explainability in Chest X-rays},
  author={TBD},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- JSRT Database: Japanese Society of Radiological Technology
- Montgomery & Shenzhen datasets: Jaeger et al., 2014
- Seg-Grad-CAM: Vinogradova et al., AAAI 2020

## Contact

For questions or collaboration inquiries, please open an issue or contact [your-email].

---

**Status**: Active Development | Phase A (Scaffolding & Data)
