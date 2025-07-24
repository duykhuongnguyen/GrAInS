import os
import argparse
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from utils.data import load_data
from utils.model import load_llm_model_and_tokenizer
from utils.steering import (
    load_hidden_states,
    compute_steering_vector,
    evaluate,
    evaluate_steering
)
from utils.config import MODEL_NAME_MAP


def run_single_eval(args, method, layer_idx, alpha, mode):
    """Evaluate model with a single steering vector configuration."""
    # Load and compute steering vector
    hidden_states = load_hidden_states(args.hidden_states_path)
    steering_vec = compute_steering_vector(hidden_states, layer_idx, method)

    if mode == "both":
        hidden_states_neg = load_hidden_states(args.hidden_states_neg_path)
        steering_vec_neg = compute_steering_vector(hidden_states_neg, layer_idx, method)
        steering_vec -= steering_vec_neg

    # Save steering vector
    os.makedirs(args.steering_vectors_dir, exist_ok=True)
    vec_filename = f"steering_{args.dataset_name}_{args.model_name}_{method}_layer{layer_idx}_alpha{alpha}_{mode}.npy"
    vec_path = os.path.join(args.steering_vectors_dir, vec_filename)
    np.save(vec_path, steering_vec)

    # Evaluate with steering
    acc, probs, labels = evaluate_steering(
        dataset=args.dataset,
        model=args.model,
        tokenizer=args.tokenizer,
        steering_vector=steering_vec,
        layer_idx=layer_idx,
        alpha=alpha
    )

    return {
        "method": method,
        "layer_idx": layer_idx,
        "alpha": alpha,
        "mode": mode,
        "accuracy": acc,
        "labels": labels,
        "probs": probs,
        "steering_vector_path": vec_path
    }


def main(args):
    print("Loading model and dataset...")
    model, tokenizer = load_llm_model_and_tokenizer(MODEL_NAME_MAP[args.model_name])
    dataset = load_data(args.dataset_name, args.num_samples, args.split)

    # Attach model components to args for convenience
    args.model = model
    args.tokenizer = tokenizer
    args.dataset = dataset

    print("Evaluating base model (no steering)...")
    acc_base, probs_base, labels_base = evaluate(dataset, model, tokenizer)
    print(f"\nBase model accuracy: {acc_base:.4f}")

    results = [{
        "method": "base_model",
        "layer_idx": "",
        "alpha": "",
        "mode": "",
        "accuracy": acc_base,
        "labels": labels_base,
        "probs": probs_base,
        "steering_vector_path": ""
    }]

    print("\nRunning steering...")

    param_grid = product(args.methods, args.layer_idxs, args.alphas, args.mode)
    for method, layer_idx, alpha, mode in tqdm(param_grid,
                                               total=len(args.methods) * len(args.layer_idxs) * len(args.alphas) * len(args.mode)):
        result = run_single_eval(args, method, layer_idx, alpha, mode)
        results.append(result)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    summary_filename = (
        f"summary_{args.dataset_name}_{args.model_name}_"
        f"{args.num_samples}_{args.attribution}_{args.top_k}_{args.grad_method}.csv"
    )
    output_path = os.path.join(args.output_dir, summary_filename)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved all results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steering LLM")

    # Dataset and model
    parser.add_argument("--dataset_name", type=str, default="truthfulqa", help="Dataset to use")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b", help="Model identifier")
    parser.add_argument("--num_samples", type=str, default="all", help="Number of samples to evaluate")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")

    # Paths
    parser.add_argument("--hidden_states_path", type=str, default="data/hidden_states", help="Path to positive hidden states")
    parser.add_argument("--hidden_states_neg_path", type=str, default="data/hidden_states_neg", help="Path to negative hidden states")
    parser.add_argument("--output_dir", type=str, default="results/steering", help="Directory to save results")
    parser.add_argument("--steering_vectors_dir", type=str, default="data/steering_vectors", help="Where to save steering vectors")

    # Attribution parameters
    parser.add_argument("--attribution", type=str, default="contrastive", choices=["normal", "contrastive"], help="Attribution mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top-k tokens for attribution")
    parser.add_argument("--grad_method", type=str, default="integrated_gradients", help="Gradient attribution method")

    # Steering vector search space
    parser.add_argument("--mode", nargs="+", default=["pos", "both"], help="Steering mode: 'pos' or 'both'")
    parser.add_argument("--methods", nargs="+", default=["pca", "mean"], help="Vector construction methods")
    parser.add_argument("--layer_idxs", nargs="+", type=int, default=[28, 29, 30, 31], help="Transformer layer indices")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="Steering vector scaling factors")

    args = parser.parse_args()
    main(args)