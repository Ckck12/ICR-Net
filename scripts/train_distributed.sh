#!/bin/bash

# ICR-Net Distributed Training Script
# Usage: bash scripts/train_distributed.sh

# Environment variables
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_HOME=$HOME/.cache/torch
export HF_HOME=$HOME/.cache/huggingface

# Create cache directories
mkdir -p $HOME/.cache/torch/hub/checkpoints
mkdir -p $HOME/.cache/huggingface

# Basic settings
CONFIG_PATH="src/configs/icr_net.yaml"
OUTPUT_DIR_BASE="./checkpoints"

# Corruption types and severities to train
CORRUPTION_TYPES=(
    "bit_error"
    "h264_crf"
    "h264_abr"
    "h265_crf"
    "h265_abr"
    "motion_blur"
    "packet_loss"
)

SEVERITIES=(3 5)

echo "====== Starting ICR-Net distributed training ======"
echo "Results will be saved in '${OUTPUT_DIR_BASE}' directory."
echo "========================================"

# Main execution loop
for corruption_type in "${CORRUPTION_TYPES[@]}"; do
    for severity in "${SEVERITIES[@]}"; do
        
        OUTPUT_PATH="${OUTPUT_DIR_BASE}/${corruption_type}_sev${severity}"

        echo ""
        echo "------------------------------------------------------------"
        echo ">>> [ICR-Net DISTRIBUTED TRAINING START]"
        echo ">>> Corruption: ${corruption_type}, Severity: ${severity}"
        echo "------------------------------------------------------------"
        
        # Distributed training execution (2 GPUs)
        CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 scripts/train.py \
            --config "$CONFIG_PATH" \
            --train_corruption "$corruption_type" \
            --train_severity "$severity" \
            --output_dir "$OUTPUT_PATH"
                
        echo ""
        echo ">>> [ICR-Net DISTRIBUTED TRAINING FINISHED]"
        echo ">>> Corruption: ${corruption_type}, Severity: ${severity}"
        echo ">>> Checkpoints and logs are saved in: ${OUTPUT_PATH}"
        echo "------------------------------------------------------------"

    done
done

echo ""
echo "====== All ICR-Net distributed training completed. ======"
