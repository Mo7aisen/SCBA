# SCBA Quick Start Guide

## Automated Pipeline (Recommended)

The easiest way to run the entire SCBA workflow is using the automated pipeline:

### 1. Start Background Pipeline

```bash
cd /home/mohaisen_mohammed/SCBA
./scripts/start_background_pipeline.sh
```

This will:
- Prepare JSRT and Montgomery datasets
- Train U-Net models on both datasets (GPU accelerated)
- Evaluate models on test sets
- Save all results to `experiments/results/`

**The pipeline runs in the background and survives terminal disconnection!**

### 2. Monitor Progress

```bash
# Check pipeline status
./scripts/monitor_pipeline.sh

# Follow logs in real-time
tail -f logs/nohup_*.out

# Check GPU usage
nvidia-smi
```

### 3. View Results

Results will be saved to:
- **Models**: `runs/jsrt_unet_baseline_*.pt`, `runs/montgomery_unet_baseline_*.pt`
- **Evaluation**: `experiments/results/jsrt_eval/`, `experiments/results/montgomery_eval/`
- **Logs**: `logs/`

---

## Manual Execution

If you prefer to run steps manually:

### Step 1: Prepare Datasets

```bash
# JSRT
python -m scba.scripts.prep_jsrt \
    --data_root /home/mohaisen_mohammed/Datasets/JSRT \
    --out data/jsrt \
    --visualize

# Montgomery
python -m scba.scripts.prep_montgomery \
    --data_root /home/mohaisen_mohammed/Datasets/Montgomery \
    --out data/montgomery \
    --visualize
```

### Step 2: Train Models

```bash
# Train on JSRT
python -m scba.train.train_seg \
    --data jsrt \
    --epochs 80 \
    --batch_size 4 \
    --amp \
    --save runs/jsrt_unet.pt

# Train on Montgomery
python -m scba.train.train_seg \
    --data montgomery \
    --epochs 80 \
    --batch_size 4 \
    --amp \
    --save runs/montgomery_unet.pt
```

### Step 3: Evaluate Models

```bash
# Evaluate JSRT
python -m scba.train.eval_seg \
    --data jsrt \
    --ckpt runs/jsrt_unet.pt \
    --visualize \
    --out eval_results/jsrt

# Evaluate Montgomery
python -m scba.train.eval_seg \
    --data montgomery \
    --ckpt runs/montgomery_unet.pt \
    --visualize \
    --out eval_results/montgomery
```

---

## Testing XAI Methods

```python
import torch
import numpy as np
from scba.models.unet import UNet
from scba.xai.common import explain
from PIL import Image

# Load model
model = UNet(n_channels=1, n_classes=2)
checkpoint = torch.load('runs/jsrt_unet.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load image
image = np.array(Image.open('path/to/image.png').convert('L'))
image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0

# Generate explanation
saliency = explain(
    image,
    model,
    method='seg_grad_cam',
    target_class=1,
    device='cuda'
)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(saliency.map, cmap='jet')
plt.show()
```

---

## Testing Counterfactual Generators

```python
import numpy as np
from scba.cf.borders import apply_border_edit
from scba.cf.gaussian_nodules import insert_gaussian_nodule
from scba.cf.inpaint import repair_nodule

# Border edit
image_cf, mask_cf, roi_band = apply_border_edit(
    image,
    mask,
    radius_px=4,
    operation='dilate',
    seed=42
)

# Nodule insertion
image_with_nodule, nodule_mask, center = insert_gaussian_nodule(
    image,
    lung_mask,
    sigma=6.0,
    delta_intensity=0.3,
    seed=42
)

# Repair
image_repaired = repair_nodule(image_with_nodule, nodule_mask)
```

---

## Computing CF Metrics

```python
from scba.metrics.cf_consistency import compute_cf_metrics

# Get saliency maps for original, perturbed, repaired
saliency_orig = explain(image_orig, model, method='seg_grad_cam')
saliency_pert = explain(image_pert, model, method='seg_grad_cam')
saliency_repair = explain(image_repair, model, method='seg_grad_cam')

# Compute metrics
metrics = compute_cf_metrics(
    saliency_orig.map,
    saliency_pert.map,
    saliency_repair.map,
    roi_band
)

print(f"ΔAM-ROI: {metrics['delta_am_roi']:.4f}")
print(f"CoA Shift: {metrics['shift_distance']:.2f} pixels")
print(f"Directional Consistency: {metrics['directional_consistency']:.2f}")
```

---

## Expected Timeline

- **Data Prep**: 5-10 minutes
- **Training (JSRT)**: 2-3 hours on GPU (A100/V100)
- **Training (Montgomery)**: 1-2 hours on GPU
- **Evaluation**: 5-10 minutes per dataset

**Total automated pipeline**: ~3-5 hours

---

## Troubleshooting

### GPU Out of Memory

Reduce batch size:
```bash
python -m scba.train.train_seg --batch_size 2 ...
```

### Dataset Not Found

Update paths in scripts or pass `--data_root`:
```bash
python -m scba.scripts.prep_jsrt --data_root /your/path/to/JSRT
```

### Check Pipeline Status

```bash
# Is it still running?
ps aux | grep python

# GPU usage
nvidia-smi

# Latest errors
tail -100 logs/pipeline_*.log
```

---

## Next Steps

After baseline training completes:

1. **Run counterfactual sweeps** (full XAI audit)
2. **Launch interactive UI** for visualization
3. **Generate paper figures** for publication

See `PROJECT_STATUS.md` for detailed phase tracking.

---

**Questions?** Check the main `README.md` or implementation plan.
