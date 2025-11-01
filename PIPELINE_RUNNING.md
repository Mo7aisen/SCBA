# 🚀 SCBA Pipeline Running

**Status**: ✅ ACTIVE
**Started**: 2025-11-01 20:32:53
**Process ID**: 604970
**GPU**: NVIDIA TITAN Xp (12GB)

---

## Current Configuration

- **Batch Size**: 2 (optimized for 12GB GPU)
- **Mixed Precision**: Enabled (AMP)
- **Learning Rate**: 0.0001
- **Optimizer**: AdamW
- **Loss**: Dice + BCE
- **Early Stopping**: 15 epochs patience

---

## Pipeline Phases

### ✅ Phase 1: Data Preparation (Complete)
- JSRT dataset prepared and split
- Montgomery dataset (skipped - needs proper structure)

### 🔄 Phase 2: JSRT Training (In Progress)
- **Expected Duration**: 2-3 hours
- **Model**: U-Net (31M parameters)
- **Target Metrics**: Dice > 0.95, IoU > 0.90

### ⏳ Phase 3: Montgomery Training (Pending)
- Will skip if dataset issues persist
- Can be run manually later if needed

### ⏳ Phase 4: Evaluation (Pending)
- Test set metrics
- Visualization generation

---

## Monitoring Commands

### Real-time Log
```bash
tail -f /home/mohaisen_mohammed/SCBA/logs/nohup_20251101_203253.out
```

### Training Log
```bash
tail -f /home/mohaisen_mohammed/SCBA/logs/train_jsrt_*.log
```

### Check Status
```bash
./scripts/monitor_pipeline.sh
```

### GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Auto-refresh every second
```

### Process Status
```bash
ps -p 604970
```

---

## Expected Output Files

### After Training Completes:

**Models**:
- `runs/jsrt_unet_baseline_20251101_203253.pt`
- `runs/jsrt_unet_baseline_20251101_203253_history.json`

**Evaluation**:
- `experiments/results/jsrt_eval/jsrt_test_metrics.json`
- `experiments/results/jsrt_eval/jsrt_test_predictions.png`

**Logs**:
- `logs/pipeline_20251101_203253.log`
- `logs/train_jsrt_20251101_203253.log`
- `logs/nohup_20251101_203253.out`

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Data Prep | 5 min | ✅ Complete |
| JSRT Training | 2-3 hours | 🔄 Running |
| Evaluation | 10 min | ⏳ Pending |
| **Total** | **~3 hours** | **~10% done** |

---

## What Happens Next

1. **Training completes** → Best model saved automatically
2. **Evaluation runs** → Test metrics and visualizations generated
3. **Pipeline finishes** → Summary printed to log

Then you can:
- View results in `experiments/results/`
- Run XAI explanations on the trained model
- Create counterfactual perturbations
- Generate paper figures

---

## If Training Fails

### Out of Memory Again?
```bash
# Kill process
kill 604970

# Edit batch size to 1
nano scripts/run_full_pipeline.sh
# Change --batch_size 2 to --batch_size 1

# Restart
./scripts/start_background_pipeline.sh
```

### Other Errors?
```bash
# Check error log
tail -100 logs/train_jsrt_*.log

# Check CUDA errors
dmesg | tail -20
```

---

## After Completion

### View Results
```bash
# Test metrics
cat experiments/results/jsrt_eval/jsrt_test_metrics.json

# Visualizations
eog experiments/results/jsrt_eval/jsrt_test_predictions.png
```

### Use Trained Model
```python
import torch
from scba.models.unet import UNet

model = UNet(n_channels=1, n_classes=2)
checkpoint = torch.load('runs/jsrt_unet_baseline_20251101_203253.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best Dice: {checkpoint['best_dice']:.4f}")
print(f"Epoch: {checkpoint['epoch']}")
```

---

**This pipeline will continue running even if you close this terminal!**

Use `./scripts/monitor_pipeline.sh` anytime to check progress.

---

*Last updated: 2025-11-01 20:33*
