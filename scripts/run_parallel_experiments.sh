#!/bin/bash
# Run SFT experiments in parallel across multiple GPUs
# Experiment 1: Varying dataset sizes {128, 256, 512, 1024, 7500}

set -e

cd /root/cs336-assignment5-alignment

# Set CUDA environment
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=/usr/lib/nvidia-cuda-toolkit/bin:$PATH

# Common parameters
BATCH_SIZE=4
LEARNING_RATE=2e-5
GRAD_ACCUM=8
NUM_EPOCHS=1
VLLM_MEM_UTIL=0.5  # Increased for faster inference
WANDB_PROJECT="sft-math-clean"

# Data paths
MODEL_PATH="/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
TRAIN_DATA="/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
VAL_DATA="/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"

# Log directory
LOG_DIR="/tmp/sft_experiments"
mkdir -p $LOG_DIR

echo "=========================================="
echo "Starting parallel SFT experiments"
echo "=========================================="
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "vLLM memory utilization: $VLLM_MEM_UTIL"
echo "=========================================="

# Function to run experiment
run_experiment() {
    local GPU=$1
    local SAMPLES=$2
    local LOG_FILE="$LOG_DIR/exp_${SAMPLES}samples_gpu${GPU}.log"
    
    echo "[GPU $GPU] Starting experiment with $SAMPLES samples -> $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m cs336_alignment.sft \
        --model_path "$MODEL_PATH" \
        --train_data_path "$TRAIN_DATA" \
        --val_data_path "$VAL_DATA" \
        --num_samples $SAMPLES \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --vllm_gpu 0 \
        --vllm_gpu_memory_utilization $VLLM_MEM_UTIL \
        --wandb_project "$WANDB_PROJECT" \
        > "$LOG_FILE" 2>&1 &
    
    echo $!
}

# Run experiments in parallel on different GPUs
# GPU 0: 128 samples
# GPU 1: 256 samples
# GPU 2: 512 samples
# GPU 3: 1024 samples
# GPU 4: 7500 samples (full)

PID0=$(run_experiment 0 128)
PID1=$(run_experiment 1 256)
PID2=$(run_experiment 2 512)
PID3=$(run_experiment 3 1024)
PID4=$(run_experiment 4 7500)

echo ""
echo "All experiments started!"
echo "PIDs: $PID0 $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/exp_*samples*.log"
echo ""
echo "Check status with:"
echo "  ps aux | grep cs336_alignment.sft"
echo ""
echo "Waiting for all experiments to complete..."

# Wait for all processes
wait $PID0 && echo "[GPU 0] 128 samples completed" &
wait $PID1 && echo "[GPU 1] 256 samples completed" &
wait $PID2 && echo "[GPU 2] 512 samples completed" &
wait $PID3 && echo "[GPU 3] 1024 samples completed" &
wait $PID4 && echo "[GPU 4] 7500 samples completed" &

wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
for f in $LOG_DIR/exp_*samples*.log; do
    samples=$(basename $f | grep -oP '\d+(?=samples)')
    accuracy=$(grep "Final Validation Accuracy" $f | grep -oP '[\d.]+(?=%)')
    echo "  $samples samples: ${accuracy}%"
done
