# Run Card Template

## Experiment Metadata

- **Run ID**: [Unique identifier]
- **Date**: [YYYY-MM-DD]
- **Researcher**: [Name]
- **Purpose**: [Brief description of experiment goal]
- **Status**: [In Progress / Completed / Failed]

## Dataset

- **Name**: [JSRT / Montgomery / Combined]
- **Split**: Train: X samples, Val: Y samples, Test: Z samples
- **Preprocessing**: [Resolution, normalization, etc.]
- **Data Root**: [Path to dataset]

## Model Configuration

- **Architecture**: [U-Net / Attention U-Net / nnU-Net]
- **Input Channels**: 1 (grayscale)
- **Output Classes**: 2 (background, lung)
- **Base Features**: 64
- **Total Parameters**: [Number of parameters]

## Training Configuration

### Hyperparameters
- **Epochs**: 80
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Weight Decay**: 1e-4
- **Loss Function**: Dice + BCE
- **LR Scheduler**: Cosine Annealing
- **Early Stopping**: 15 epochs patience

### Augmentation
- Horizontal flip: p=0.5
- Rotation: ±10°
- Brightness/Contrast: ±0.2
- CLAHE: p=0.3
- Gaussian Noise: p=0.2

### Reproducibility
- **Seed**: 42
- **CUDA Version**: [Version]
- **PyTorch Version**: [Version]
- **Mixed Precision**: Enabled / Disabled

## Compute Resources

- **GPU**: [Model and VRAM]
- **Training Time**: [Hours]
- **GPU Memory Usage**: [Peak memory in GB]
- **CPU**: [Model and cores]
- **RAM**: [Total RAM]

## Results

### Quantitative Metrics (Test Set)

| Metric | Value | Std Dev |
|--------|-------|---------|
| Dice Coefficient | 0.XXX | ±0.XXX |
| IoU | 0.XXX | ±0.XXX |
| Boundary F-Score | 0.XXX | ±0.XXX |
| Pixel Accuracy | 0.XXX | ±0.XXX |
| Sensitivity | 0.XXX | ±0.XXX |
| Specificity | 0.XXX | ±0.XXX |

### Training Curves
- [Link to training loss/metric plots]

### Best Epoch
- **Epoch**: [Number]
- **Val Dice**: [Score]

## Artifacts

- **Model Checkpoint**: [Path to .pt file]
- **Training History**: [Path to history.json]
- **Visualizations**: [Path to prediction samples]
- **Config File**: [Path to .yaml]

## Notes & Observations

- [Any notable behaviors during training]
- [Convergence patterns]
- [Failure cases observed]
- [Comparison to previous runs]

## Reproducibility Checklist

- [ ] Seed set for all random generators
- [ ] Data splits documented
- [ ] Config file saved
- [ ] Model checkpoint saved
- [ ] Training history logged
- [ ] Environment dependencies documented

## Next Steps

- [Planned follow-up experiments]
- [Modifications to try]
- [Issues to investigate]

---

**Signature**: [Name] | **Date**: [YYYY-MM-DD]
