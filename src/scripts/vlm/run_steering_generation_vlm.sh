#!/bin/bash

# Set default values
DATASET_NAME="spa-vl"
MODEL_NAME="llava-1.6-7b"  
NUM_SAMPLES="all"
SPLIT="validation"

# Attribution and steering config
NUM_SAMPLES_STEER=50
ATTRIBUTION="contrastive"
GRAD_METHOD="integrated_gradients"  # vanilla 
K=50

# Paths to hidden states
HIDDEN_STATES_PATH="data/vlm/hidden_states/hidden_states_${DATASET_NAME}_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_pos_${GRAD_METHOD}.npz"
HIDDEN_STATES_NEG_PATH="data/vlm/hidden_states/hidden_states_${DATASET_NAME}_${MODEL_NAME}_${NUM_SAMPLES_STEER}_${ATTRIBUTION}_${K}_neg_${GRAD_METHOD}.npz"

# Parameters
MODES=("pos")
METHODS=("pca")
LAYER_IDXS=(31) # 27
ALPHAS=(10.0)

# Output directory
OUTPUT_DIR="results/vlm/generation_steering"
STEERING_VECTOR_DIR="vlm/llm/steering_vectors"

# Generation config
TEMPERATURE=0.1
MAX_NEW_TOKENS=256

# Run the script
python steering_generation_vlm.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --num_samples "$NUM_SAMPLES" \
  --split "$SPLIT" \
  --hidden_states_path "$HIDDEN_STATES_PATH" \
  --hidden_states_neg_path "$HIDDEN_STATES_NEG_PATH" \
  --top_k "$K" \
  --grad_method "$GRAD_METHOD" \
  --attribution "$ATTRIBUTION" \
  --mode "${MODES[@]}" \
  --methods "${METHODS[@]}" \
  --layer_idxs "${LAYER_IDXS[@]}" \
  --alphas "${ALPHAS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --steering_vectors_dir "$STEERING_VECTOR_DIR" \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS"