#!/bin/bash
# ============================================================================
# Qwen3-VL Multi-Task Regression Training Script
#
# This script trains a Qwen3-VL-Embedding model for multi-task regression with:
# - Point cloud feature encoding (512-dim input)
# - Cross-Attention fusion between point cloud and image features
# - Multiple parallel regression heads
# - Weighted MSE loss for task balancing
#
# Usage:
#   bash qwen3_vl_regression.sh
#
# Prerequisites:
#   - Dataset prepared in the expected format (see qwen_regression.py for format)
#   - GPU with sufficient memory (recommended: 4x A100 80GB or equivalent)
#   - transformers>=4.57, qwen_vl_utils>=0.0.14
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_TYPE="qwen3_vl_multi_regression"
MODEL_PATH="/mnt/cpfs/xinxuan/Qwen3-VL-Embedding-2B/"  # or Qwen/Qwen3-VL-Embedding-8B

# Output directory (with timestamp)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/qwen3_svl_3tasks_${TIMESTAMP}"

# Dataset configuration
# Option 1: Use registered dataset name
# DATASET="/home/STCC-LM/dataset/preprocess/dataset_CE_C_Real_TL_S_Sim_260222.jsonl"
DATASET="multimodal_regression_local"
# Option 2: Use local dataset
# DATASET="/path/to/your/data"

# Number of regression tasks and their weights
NUM_TASKS=3
# Task weights: higher weight = more important task
# Format: comma-separated list of floats
TASK_WEIGHTS="1.0,1.0,1.0"

# Point cloud feature dimension
POINT_CLOUD_DIM=512

# GPU configuration
# CUDA_VISIBLE_DEVICES=0,1,2,3  # Uncomment and set if needed
NPROC_PER_NODE=1

# Training hyperparameters
LEARNING_RATE=1e-5
NUM_EPOCHS=20
BATCH_SIZE=4  # Per device
GRAD_ACCUM_STEPS=8
MAX_LENGTH=65536

# LoRA configuration
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# DeepSpeed configuration
# Options: zero2, zero3, or path to custom config
DEEPSPEED="zero3"

# ============================================================================
# Training Command
# ============================================================================

echo "=========================================="
echo "Qwen3-VL Multi-Task Regression Training"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Model Type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET}"
echo "Number of Tasks: ${NUM_TASKS}"
echo "Task Weights: ${TASK_WEIGHTS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Check if dataset exists (for local datasets)
if [[ "${DATASET}" == /* ]]; then
    if [ ! -e "${DATASET}" ]; then
        echo "Error: Dataset directory not found: ${DATASET}"
        exit 1
    fi
    echo "Using local dataset: ${DATASET}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training
swift sft \
    --model_type ${MODEL_TYPE} \
    --model ${MODEL_PATH} \
    \
    `# === Task Configuration ===` \
    --task_type seq_cls \
    --problem_type regression \
    --num_labels ${NUM_TASKS} \
    --task_weights ${TASK_WEIGHTS} \
    --point_cloud_dim ${POINT_CLOUD_DIM} \
    \
    `# === Tuner Configuration ===` \
    --tuner_type full \
    \
    `# === Dataset Configuration ===` \
    --dataset ${DATASET} \
    --split_dataset_ratio 0.05 \
    --remove_unused_columns false \
    --streaming false \
    --load_from_cache_file true \
    --dataset_num_proc 4 \
    \
    `# === Model Configuration ===` \
    --template qwen3_vl_emb \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --max_length ${MAX_LENGTH} \
    --padding_free true \
    \
    `# === Training Configuration ===` \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    `# === Evaluation & Checkpoints ===` \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    \
    `# === Output Configuration ===` \
    --output_dir ${OUTPUT_DIR} \
    --run_name qwen3_vl_regression_${TIMESTAMP} \
    --report_to tensorboard \
    \
    `# === Performance Configuration ===` \
    --dataloader_num_workers  \
    --gradient_checkpointing true \
    --optim adamw_torch \
    --bf16 true \
    --tf32 true \
    \
    `# === DeepSpeed ===` \
    --deepspeed ${DEEPSPEED} \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Output saved to: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed. Check the log for details:"
    echo "${OUTPUT_DIR}/training.log"
    echo "=========================================="
    exit 1
fi

