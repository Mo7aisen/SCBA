#!/bin/bash
################################################################################
# Monitor SCBA Pipeline Progress
################################################################################

PROJECT_ROOT="/home/mohaisen_mohammed/SCBA"
LOG_DIR="${PROJECT_ROOT}/logs"

# Find most recent pipeline log
LATEST_LOG=$(ls -t "${LOG_DIR}"/pipeline_*.log 2>/dev/null | head -1)
LATEST_NOHUP=$(ls -t "${LOG_DIR}"/nohup_*.out 2>/dev/null | head -1)

echo "========================================="
echo "SCBA Pipeline Monitor"
echo "========================================="
echo ""

# Check if pipeline is running
LATEST_PID_FILE=$(ls -t "${LOG_DIR}"/pipeline_*.pid 2>/dev/null | head -1)
if [ -n "${LATEST_PID_FILE}" ]; then
    PID=$(cat "${LATEST_PID_FILE}")
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "✓ Pipeline is RUNNING (PID: ${PID})"
    else
        echo "✗ Pipeline has FINISHED or STOPPED"
    fi
else
    echo "⚠ No pipeline PID file found"
fi

echo ""
echo "Most recent logs:"
echo "  Main: ${LATEST_LOG}"
echo "  Nohup: ${LATEST_NOHUP}"
echo ""

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    echo "----------------------------------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
fi

# Show latest log tail
if [ -n "${LATEST_LOG}" ]; then
    echo "Latest Log Tail:"
    echo "----------------------------------------"
    tail -20 "${LATEST_LOG}"
fi

echo ""
echo "========================================="
echo "To follow log in real-time:"
if [ -n "${LATEST_NOHUP}" ]; then
    echo "  tail -f ${LATEST_NOHUP}"
else
    echo "  tail -f ${LATEST_LOG}"
fi
echo "========================================="
