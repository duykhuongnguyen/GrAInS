#!/bin/bash

# Default parameters
DATASET_NAME="truthfulqa"
MODEL_NAME="llama-3.1-8b-instruct"  # qwen-2.5-7b-instruct
NUM_SAMPLES=50
ATTRIBUTION="contrastive"
GRAD_METHOD="vanilla"  # integrated_gradients 
SPLIT="validation"
OUTPUT_DIR_META="results/llm/token_attribution"
OUTPUT_DIR_HIDDEN="data/llm/hidden_states"

# Parameters
TOP_K_VALUES=(10)
TOP_VALUES=("pos" "neg")

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR_META
mkdir -p $OUTPUT_DIR_HIDDEN

# Loop through combinations and run script
for TOP_K in "${TOP_K_VALUES[@]}"; do
  for TOP in "${TOP_VALUES[@]}"; do
    echo "Running with TOP_K=$TOP_K and TOP=$TOP"
    
    python token_ablation.py \
      --dataset_name $DATASET_NAME \
      --model_name $MODEL_NAME \
      --num_samples $NUM_SAMPLES \
      --top_k $TOP_K \
      --top $TOP \
      --attribution $ATTRIBUTION \
      --grad_method $GRAD_METHOD \
      --split $SPLIT \
      --output_path_meta "${OUTPUT_DIR_META}" \
      --output_path_hidden "${OUTPUT_DIR_HIDDEN}"
  done
done