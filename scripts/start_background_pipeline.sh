#!/bin/bash
################################################################################
# Start SCBA Pipeline in Background
#
# Launches full pipeline using nohup to survive terminal disconnect
################################################################################

PROJECT_ROOT="/home/mohaisen_mohammed/SCBA"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"

echo "========================================="
echo "Starting SCBA Pipeline in Background"
echo "========================================="
echo ""
echo "The pipeline will run these phases:"
echo "  1. Data preparation (JSRT & Montgomery)"
echo "  2. Train U-Net on JSRT (~2-3 hours)"
echo "  3. Train U-Net on Montgomery (~1-2 hours)"
echo "  4. Evaluate models on test sets"
echo ""
echo "Logs will be saved to: ${LOG_DIR}"
echo ""

# Run in background with nohup
nohup "${PROJECT_ROOT}/scripts/run_full_pipeline.sh" \
    > "${LOG_DIR}/nohup_${TIMESTAMP}.out" 2>&1 &

PID=$!

echo "✓ Pipeline started in background"
echo "  Process ID: ${PID}"
echo "  Main log: ${LOG_DIR}/nohup_${TIMESTAMP}.out"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOG_DIR}/nohup_${TIMESTAMP}.out"
echo ""
echo "Check if still running:"
echo "  ps -p ${PID}"
echo ""
echo "View GPU usage:"
echo "  nvidia-smi"
echo ""

# Save PID for later
echo ${PID} > "${LOG_DIR}/pipeline_${TIMESTAMP}.pid"

echo "PID saved to: ${LOG_DIR}/pipeline_${TIMESTAMP}.pid"
echo "========================================="
