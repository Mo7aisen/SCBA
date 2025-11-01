#!/bin/bash
################################################################################
# SCBA Full Pipeline - Automated Execution
#
# Runs complete SCBA workflow: data prep → training → evaluation → experiments
# Uses nohup to run in background, survives terminal disconnect
################################################################################

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/home/mohaisen_mohammed/SCBA"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_DIR="${PROJECT_ROOT}/runs"
RESULTS_DIR="${PROJECT_ROOT}/experiments/results"

# Create directories
mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${RESULTS_DIR}"

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

echo "========================================" | tee -a "${MAIN_LOG}"
echo "SCBA Full Pipeline Started" | tee -a "${MAIN_LOG}"
echo "Time: $(date)" | tee -a "${MAIN_LOG}"
echo "========================================" | tee -a "${MAIN_LOG}"

# Change to project root
cd "${PROJECT_ROOT}"

################################################################################
# Phase 1: Data Preparation
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "[Phase 1/5] Data Preparation" | tee -a "${MAIN_LOG}"
echo "----------------------------------------" | tee -a "${MAIN_LOG}"

# JSRT
echo "Preparing JSRT dataset..." | tee -a "${MAIN_LOG}"
python -m scba.scripts.prep_jsrt \
    --data_root /home/mohaisen_mohammed/Datasets/JSRT \
    --out data/jsrt \
    --visualize \
    >> "${MAIN_LOG}" 2>&1 || echo "JSRT prep failed (may not exist)" | tee -a "${MAIN_LOG}"

# Montgomery
echo "Preparing Montgomery dataset..." | tee -a "${MAIN_LOG}"
python -m scba.scripts.prep_montgomery \
    --data_root /home/mohaisen_mohammed/Datasets/Montgomery \
    --out data/montgomery \
    --visualize \
    >> "${MAIN_LOG}" 2>&1 || echo "Montgomery prep failed (may not exist)" | tee -a "${MAIN_LOG}"

echo "✓ Data preparation complete" | tee -a "${MAIN_LOG}"

################################################################################
# Phase 2: Baseline Training (JSRT)
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "[Phase 2/5] Baseline Model Training - JSRT" | tee -a "${MAIN_LOG}"
echo "----------------------------------------" | tee -a "${MAIN_LOG}"

if [ -f "/home/mohaisen_mohammed/Datasets/JSRT/images" ] || [ -d "/home/mohaisen_mohammed/Datasets/JSRT/images" ]; then
    echo "Training U-Net on JSRT..." | tee -a "${MAIN_LOG}"
    # Set CUDA memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python -m scba.train.train_seg \
        --data jsrt \
        --data_root /home/mohaisen_mohammed/Datasets/JSRT \
        --arch unet \
        --epochs 80 \
        --batch_size 2 \
        --lr 0.0001 \
        --loss dice_bce \
        --amp \
        --early_stop 15 \
        --save "${RUN_DIR}/jsrt_unet_baseline_${TIMESTAMP}.pt" \
        --save_dir "${RUN_DIR}" \
        --seed 42 \
        --num_workers 2 \
        >> "${LOG_DIR}/train_jsrt_${TIMESTAMP}.log" 2>&1 &

    TRAIN_JSRT_PID=$!
    echo "JSRT training started (PID: ${TRAIN_JSRT_PID})" | tee -a "${MAIN_LOG}"

    # Wait for training to complete
    wait ${TRAIN_JSRT_PID}
    echo "✓ JSRT training complete" | tee -a "${MAIN_LOG}"
else
    echo "⚠ JSRT dataset not found, skipping training" | tee -a "${MAIN_LOG}"
fi

################################################################################
# Phase 3: Baseline Training (Montgomery)
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "[Phase 3/5] Baseline Model Training - Montgomery" | tee -a "${MAIN_LOG}"
echo "----------------------------------------" | tee -a "${MAIN_LOG}"

if [ -f "/home/mohaisen_mohammed/Datasets/Montgomery/CXR_png" ] || [ -d "/home/mohaisen_mohammed/Datasets/Montgomery/CXR_png" ]; then
    echo "Training U-Net on Montgomery..." | tee -a "${MAIN_LOG}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python -m scba.train.train_seg \
        --data montgomery \
        --data_root /home/mohaisen_mohammed/Datasets/Montgomery \
        --arch unet \
        --epochs 80 \
        --batch_size 2 \
        --lr 0.0001 \
        --loss dice_bce \
        --amp \
        --early_stop 15 \
        --save "${RUN_DIR}/montgomery_unet_baseline_${TIMESTAMP}.pt" \
        --save_dir "${RUN_DIR}" \
        --seed 42 \
        --num_workers 2 \
        >> "${LOG_DIR}/train_montgomery_${TIMESTAMP}.log" 2>&1 &

    TRAIN_MONT_PID=$!
    echo "Montgomery training started (PID: ${TRAIN_MONT_PID})" | tee -a "${MAIN_LOG}"

    # Wait for training
    wait ${TRAIN_MONT_PID}
    echo "✓ Montgomery training complete" | tee -a "${MAIN_LOG}"
else
    echo "⚠ Montgomery dataset not found, skipping training" | tee -a "${MAIN_LOG}"
fi

################################################################################
# Phase 4: Model Evaluation
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "[Phase 4/5] Model Evaluation" | tee -a "${MAIN_LOG}"
echo "----------------------------------------" | tee -a "${MAIN_LOG}"

# Find trained models
JSRT_MODEL=$(ls -t "${RUN_DIR}"/jsrt_unet_baseline_*.pt 2>/dev/null | head -1 || echo "")
MONT_MODEL=$(ls -t "${RUN_DIR}"/montgomery_unet_baseline_*.pt 2>/dev/null | head -1 || echo "")

if [ -n "${JSRT_MODEL}" ]; then
    echo "Evaluating JSRT model: ${JSRT_MODEL}" | tee -a "${MAIN_LOG}"
    python -m scba.train.eval_seg \
        --data jsrt \
        --ckpt "${JSRT_MODEL}" \
        --split test \
        --visualize \
        --n_vis 10 \
        --out "${RESULTS_DIR}/jsrt_eval" \
        >> "${LOG_DIR}/eval_jsrt_${TIMESTAMP}.log" 2>&1
    echo "✓ JSRT evaluation complete" | tee -a "${MAIN_LOG}"
fi

if [ -n "${MONT_MODEL}" ]; then
    echo "Evaluating Montgomery model: ${MONT_MODEL}" | tee -a "${MAIN_LOG}"
    python -m scba.train.eval_seg \
        --data montgomery \
        --ckpt "${MONT_MODEL}" \
        --split test \
        --visualize \
        --n_vis 10 \
        --out "${RESULTS_DIR}/montgomery_eval" \
        >> "${LOG_DIR}/eval_montgomery_${TIMESTAMP}.log" 2>&1
    echo "✓ Montgomery evaluation complete" | tee -a "${MAIN_LOG}"
fi

################################################################################
# Phase 5: Summary
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "[Phase 5/5] Pipeline Summary" | tee -a "${MAIN_LOG}"
echo "----------------------------------------" | tee -a "${MAIN_LOG}"

echo "" | tee -a "${MAIN_LOG}"
echo "========================================" | tee -a "${MAIN_LOG}"
echo "SCBA Pipeline Complete!" | tee -a "${MAIN_LOG}"
echo "Time: $(date)" | tee -a "${MAIN_LOG}"
echo "========================================" | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"
echo "Logs saved to: ${LOG_DIR}" | tee -a "${MAIN_LOG}"
echo "Models saved to: ${RUN_DIR}" | tee -a "${MAIN_LOG}"
echo "Results saved to: ${RESULTS_DIR}" | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"

# Print summary of results
if [ -f "${RESULTS_DIR}/jsrt_eval/jsrt_test_metrics.json" ]; then
    echo "JSRT Test Results:" | tee -a "${MAIN_LOG}"
    cat "${RESULTS_DIR}/jsrt_eval/jsrt_test_metrics.json" | tee -a "${MAIN_LOG}"
fi

if [ -f "${RESULTS_DIR}/montgomery_eval/montgomery_test_metrics.json" ]; then
    echo "Montgomery Test Results:" | tee -a "${MAIN_LOG}"
    cat "${RESULTS_DIR}/montgomery_eval/montgomery_test_metrics.json" | tee -a "${MAIN_LOG}"
fi

echo "" | tee -a "${MAIN_LOG}"
echo "Full log: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "========================================" | tee -a "${MAIN_LOG}"
