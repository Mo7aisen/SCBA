# SCBA Project Status Report

**Generated**: 2025-11-01
**Status**: Phase A & B Complete (Week 1-2 of 9)

---

## 🎯 Project Overview

**Synthetic Counterfactual Border Audit (SCBA)** is a systematic framework for auditing explainability methods in medical image segmentation. The project focuses on chest X-ray lung segmentation and introduces novel counterfactual evaluation metrics.

### Key Innovation
We synthesize **controlled, realistic perturbations** at organ borders and insert synthetic lesions to audit whether XAI explanations **follow causal edits** in the image.

---

## ✅ Completed Phases

### Phase A: Scaffolding & Data Infrastructure (Week 1) ✓

#### Repository Structure
- ✅ Professional Python package layout (`scba/`)
- ✅ Pre-commit hooks (black, isort, flake8, mypy)
- ✅ GitHub Actions CI pipeline
- ✅ pytest with coverage configuration
- ✅ Comprehensive `.gitignore` and documentation

#### Dependencies & Environment
- ✅ `environment.yml` with:
  - PyTorch 2.1.0 + CUDA 11.8
  - Captum, Zennit, SHAP, LIME (XAI libraries)
  - Albumentations, Kornia (augmentation)
  - PyDICOM, SimpleITK (medical imaging)
  - Streamlit, Gradio, FastAPI (UI frameworks)
  - W&B, MLflow (experiment tracking)
- ✅ `requirements.txt` for pip-only installations

#### Data Loaders
- ✅ **JSRT Dataset Loader**
  - 247 PA chest X-rays (2048×2048)
  - Deterministic train/val/test splits (70/15/15)
  - Left/right lung masks with automatic combination
  - Checksum verification
- ✅ **Montgomery Dataset Loader**
  - 138 PA chest X-rays with TB screening
  - Handles `_0` and `_1` mask suffixes
  - Deterministic splits with metadata CSV
- ✅ Albumentations-based augmentation pipeline
  - Geometric: flip, rotate, shift-scale
  - Intensity: brightness, contrast, CLAHE
  - Noise injection with configurable probability

#### Data Preparation Scripts
- ✅ `prep_jsrt.py`: Analysis, splits, visualization
- ✅ `prep_montgomery.py`: Analysis, splits, visualization
- ✅ `create_sanity_dataset.py`: Small subset for CI/demo

### Phase B: Baseline Segmentation Models (Week 2) ✓

#### U-Net Implementation
- ✅ Standard U-Net architecture
  - Encoder-decoder with skip connections
  - Configurable base features (default: 64)
  - Bilinear or transposed conv upsampling
  - ~31M parameters at base=64
- ✅ `get_encoder_features()` method for XAI integration
- ✅ Support for grayscale (1-channel) and RGB inputs

#### Loss Functions
- ✅ Dice Loss
- ✅ Combined Dice + BCE Loss
- ✅ Focal Loss (for class imbalance)
- ✅ Tversky Loss (generalized Dice)
- ✅ Factory function: `get_loss(name, **kwargs)`

#### Evaluation Metrics
- ✅ Dice Coefficient
- ✅ IoU (Intersection over Union)
- ✅ Boundary F-Score (contour accuracy)
- ✅ Pixel Accuracy
- ✅ Sensitivity & Specificity
- ✅ `MetricTracker` class for training monitoring

#### Training Pipeline
- ✅ `train_seg.py`: Full training script
  - Mixed precision (AMP) support
  - AdamW optimizer with cosine LR scheduling
  - Early stopping (patience=15)
  - Automatic best model saving
  - Training history logging (JSON)
- ✅ `eval_seg.py`: Evaluation CLI
  - Per-sample metric computation
  - Statistical aggregation (mean, std)
  - Visualization of predictions
  - Overlay generation (GT vs Pred)

#### Configuration & Documentation
- ✅ YAML configs for JSRT and Montgomery baselines
- ✅ Run card template with reproducibility checklist
- ✅ Expected performance benchmarks documented

---

## 📊 Current Capabilities

### You can now:
1. **Load and preprocess** JSRT and Montgomery datasets
2. **Train** U-Net models with:
   - Automatic mixed precision
   - Early stopping
   - Cosine LR scheduling
   - Multiple loss functions
3. **Evaluate** models with 6+ metrics including BF-score
4. **Visualize** predictions with GT/Pred overlays
5. **Track** experiments with deterministic seeds

### Example Usage

```bash
# Prepare datasets
python -m scba.scripts.prep_jsrt --visualize
python -m scba.scripts.prep_montgomery --visualize

# Train baseline model
python -m scba.train.train_seg \
  --data jsrt \
  --epochs 80 \
  --amp \
  --save runs/jsrt_unet_baseline.pt

# Evaluate
python -m scba.train.eval_seg \
  --data jsrt \
  --ckpt runs/jsrt_unet_baseline.pt \
  --visualize \
  --out eval_results/
```

---

## 🚧 Remaining Work (7 weeks)

### Phase C: XAI Methods (Week 3) 🔄 In Progress
**Goal**: Implement 12+ explainability methods for segmentation

- [ ] **Gradient-based**:
  - [ ] Seg-Grad-CAM
  - [ ] Seg-XRes-CAM
  - [ ] HiResCAM
  - [ ] Grad-CAM++
  - [ ] Guided Grad-CAM
  - [ ] Integrated Gradients (via Captum)
  - [ ] LRP (via Zennit)

- [ ] **Perturbation-based**:
  - [ ] Occlusion
  - [ ] RISE (Random Input Sampling)
  - [ ] LIME (superpixel-based)
  - [ ] SHAP (kernel SHAP)
  - [ ] D-RISE (optional, for detection)

- [ ] Unified API: `explain(image, model, target_mask, method)`
- [ ] Sanity check utilities (parameter/input randomization)
- [ ] Unit tests for determinism and shape correctness

### Phase D: Counterfactual Generators (Week 4)
**Goal**: Realistic, controlled border edits and lesion insertion

- [ ] **Border Edits**:
  - [ ] Morphological ops (dilation, erosion, open, close)
  - [ ] Contour jittering with Fourier noise
  - [ ] Thin-Plate Spline (TPS) warping
  - [ ] Poisson blending for seamless integration
  - [ ] ROI band extraction (symmetric contour band)

- [ ] **Gaussian Nodule Surrogates**:
  - [ ] Anisotropic 2D Gaussian blobs
  - [ ] Intensity matching to local lung window
  - [ ] Smart placement (parenchyma only, avoid borders)
  - [ ] Multi-scale Laplacian blending

- [ ] **Repair Operations**:
  - [ ] Inpainting (Telea, Navier-Stokes)
  - [ ] TPS inverse warping
  - [ ] Reversibility validation (SSIM checks)

### Phase E: Metrics & Sweeps (Week 5)
**Goal**: Novel counterfactual consistency metrics

- [ ] **Faithfulness**: Deletion/Insertion AUC with in-distribution reveals
- [ ] **CF Consistency**:
  - [ ] AM-ROI (attribution mass in ROI)
  - [ ] ΔAM-ROI (change in attribution mass)
  - [ ] CoA-Δ (center of attribution shift)
  - [ ] Directional Consistency
- [ ] **Stability**: SSIM, Pearson under noise/blur
- [ ] **Localization**: IoU, pointing game
- [ ] Sweep script: loops datasets × methods × edits
- [ ] Bootstrap confidence intervals (BCa)
- [ ] W&B/MLflow integration

### Phase F: Robustness Suite (Week 6)
**Goal**: Stress-test explanations beyond counterfactuals

- [ ] Benign perturbations: noise σ, JPEG compression q, blur r
- [ ] Domain shift: train Montgomery → test JSRT
- [ ] Adversarial: PGD/FGSM to move saliency off-ROI
- [ ] Sanity checks: Adebayo randomization tests

### Phase G: UI & Clinical Deployment (Week 7)
**Goal**: Interactive triptych viewer + PACS integration

- [ ] **Triptych UI**: Original | Perturbed | Repaired
  - [ ] Synchronized pan/zoom
  - [ ] Method switcher (dropdown)
  - [ ] Live AM-ROI / CoA-Δ display
  - [ ] Opacity sliders for overlays
- [ ] **Study Mode**: Randomized case presentation for radiologist evaluation
- [ ] **DICOM Export**:
  - [ ] Segmentation as DICOM SEG
  - [ ] Metrics as DICOM SR
  - [ ] Heatmaps as Secondary Capture
  - [ ] OHIF viewer integration

### Phase H: User Study Pack (Week 8)
- [ ] IRB templates & consent forms
- [ ] Study protocol JSON (case sets, tasks)
- [ ] Analysis notebook (trust calibration, preference stats)
- [ ] Anonymization pipeline

### Phase I: Paper Kit & Release (Week 9)
- [ ] `make_tables.py`: Generate paper figures
- [ ] Triptych panels, AUC curves, ΔAM-ROI bars
- [ ] Stability plots, domain shift charts
- [ ] README & docs site (Quickstart, Method Card, Limitations)
- [ ] Release weights, configs, scripts

---

## 📈 Progress Metrics

| Phase | Status | Completion | Files | Tests |
|-------|--------|-----------|-------|-------|
| A: Scaffolding | ✅ Complete | 100% | 15 | 2 |
| B: Models | ✅ Complete | 100% | 8 | 0 |
| C: XAI | 🔄 In Progress | 0% | 0 | 0 |
| D: CF Gen | ⏳ Pending | 0% | 0 | 0 |
| E: Metrics | ⏳ Pending | 0% | 0 | 0 |
| F: Robustness | ⏳ Pending | 0% | 0 | 0 |
| G: UI | ⏳ Pending | 0% | 0 | 0 |
| H: Study | ⏳ Pending | 0% | 0 | 0 |
| I: Paper | ⏳ Pending | 0% | 0 | 0 |
| **Overall** | **22%** | **2/9 weeks** | **23** | **2** |

---

## 🎓 Academic Quality Checklist

### Code Quality ✅
- [x] Type hints for all functions
- [x] Docstrings (NumPy style)
- [x] Pre-commit hooks enforced
- [x] Modular, extensible design
- [x] Professional logging

### Reproducibility ✅
- [x] Seed control (random, numpy, torch)
- [x] Deterministic splits with CSVs
- [x] Config files (YAML)
- [x] Run cards with metadata
- [x] Environment pinned (conda)

### Scientific Rigor
- [x] Boundary F-score (beyond Dice/IoU)
- [ ] Bootstrap CIs for statistical significance
- [ ] Sanity checks for XAI methods
- [ ] Ablation studies documented

### Clinical Readiness
- [ ] DICOM SEG/SR export
- [ ] OHIF/PACS integration
- [ ] Explanation stability indicators
- [ ] Audit logging

---

## 💡 Next Immediate Steps

1. **Start Phase C** (XAI Implementation):
   - Implement `scba/xai/common.py` with unified API
   - Add Seg-Grad-CAM as first method
   - Create unit tests for shape/dtype/determinism

2. **Train Baseline Models**:
   - Run training on JSRT (GPU required)
   - Run training on Montgomery
   - Document run cards with actual metrics

3. **Validate Data Loaders**:
   - Run `prep_jsrt.py --visualize`
   - Run `prep_montgomery.py --visualize`
   - Inspect splits.csv for both datasets

---

## 🔗 Key Files

| Component | Path |
|-----------|------|
| Main README | `README.md` |
| Implementation Plan | `synthetic_counterfactual_border_audit_scba_full_implementation_plan.md` |
| JSRT Loader | `scba/data/loaders/jsrt.py` |
| Montgomery Loader | `scba/data/loaders/montgomery.py` |
| U-Net Model | `scba/models/unet.py` |
| Training Script | `scba/train/train_seg.py` |
| Evaluation Script | `scba/train/eval_seg.py` |
| Environment | `env/environment.yml` |
| Tests | `tests/test_data_loaders.py` |

---

## 🤝 Contributing

This is an academic research project. Key design decisions:
- **No hardcoded paths** (except defaults in argparse)
- **Deterministic by default** (seed=42)
- **GPU-first** but CPU-compatible
- **Metric-rich** (never trust a single number)

---

**Last Updated**: 2025-11-01
**Next Milestone**: Phase C complete (Week 3)
**Target Submission**: MICCAI 2025 or TMI

🤖 *This project status was generated automatically by Claude Code.*
