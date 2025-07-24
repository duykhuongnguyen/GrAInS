#!/bin/bash

# Default parameters
DATASET_NAME="spa-vl"
MODEL_NAME="llava-1.6-7b"  # qwen-2.5-vl-7b-instruct
NUM_SAMPLES=50
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"  # vanilla 
SPLIT="validation"
OUTPUT_DIR_META="results/vlm/token_attribution"
OUTPUT_DIR_HIDDEN="data/vlm/hidden_states"

# Parameters
TOP_K_VALUES=(20)
TOP_VALUES=("pos" "neg")

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR_META
mkdir -p $OUTPUT_DIR_HIDDEN

# Loop through combinations and run script
for TOP_K in "${TOP_K_VALUES[@]}"; do
  for TOP in "${TOP_VALUES[@]}"; do
    echo "Running with TOP_K=$TOP_K and TOP=$TOP"
    
    python token_ablation_vlm.py \
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