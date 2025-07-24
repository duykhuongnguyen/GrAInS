import os
import argparse
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data import load_data
from utils.model import load_vlm_model_and_processor
from utils.steering import (
    load_hidden_states,
    compute_steering_vector,
    generate_answer_dataset_vlm,
    steer_and_generate_dataset_vlm,
)
from utils.config import MODEL_NAME_MAP


def run_single_eval(args, method, layer_idx, alpha, mode, base_outputs):
    """
    Run a single steering vector evaluation for VLMs and return generation results.
    """
    # Compute steering vector(s)
    hidden_states = load_hidden_states(args.hidden_states_path)
    steering_vec = compute_steering_vector(hidden_states, layer_idx, method)

    if mode == "both":
        hidden_states_neg = load_hidden_states(args.hidden_states_neg_path)
        steering_vec_neg = compute_steering_vector(hidden_states_neg, layer_idx, method)
        steering_vec -= steering_vec_neg

    # Save vector
    os.makedirs(args.steering_vectors_dir, exist_ok=True)
    vec_filename = (
        f"steering_{args.dataset_name}_{args.model_name}_{method}_"
        f"layer{layer_idx}_alpha{alpha}_{mode}.npy"
    )
    vec_path = os.path.join(args.steering_vectors_dir, vec_filename)
    np.save(vec_path, steering_vec)

    # Generate steered responses
    steered_outputs = steer_and_generate_dataset_vlm(
        dataset=args.dataset,
        model=args.model,
        processor=args.processor,
        steering_vec=steering_vec,
        layer_idx=layer_idx,
        alpha=alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    results = []
    for example, steered_output in zip(args.dataset, steered_outputs):
        question = example["question"].strip()
        true_answer = example["chosen"].strip()
        prompt_key = question
        base_output = base_outputs[prompt_key]

        results.append({
            "question": question,
            "true_answer": true_answer,
            "base_output": base_output,
            "steered_output": steered_output,
            "method": method,
            "layer_idx": layer_idx,
            "alpha": alpha,
            "mode": mode,
            "steering_vector_path": vec_path,
        })

    return results


def main(args):
    print("Loading model and dataset...")
    model, processor = load_vlm_model_and_processor(MODEL_NAME_MAP[args.model_name])
    dataset = load_data(args.dataset_name, args.num_samples, args.split)

    args.model = model
    args.processor = processor
    args.dataset = dataset

    # Generate base completions
    print("Generating base outputs...")
    base_generations = generate_answer_dataset_vlm(
        dataset, model, processor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    base_outputs = {
        ex["question"].strip(): output
        for ex, output in zip(dataset, base_generations)
    }

    all_results = []

    print("Running generation with steering vectors...")
    search_space = product(args.methods, args.layer_idxs, args.alphas, args.mode)
    total_combinations = len(args.methods) * len(args.layer_idxs) * len(args.alphas) * len(args.mode)

    for method, layer_idx, alpha, mode in tqdm(search_space, total=total_combinations):
        results = run_single_eval(args, method, layer_idx, alpha, mode, base_outputs)
        all_results.extend(results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = (
        f"generation_results_{args.dataset_name}_{args.model_name}_"
        f"{args.num_samples}_{args.attribution}_{args.top_k}_{args.grad_method}.csv"
    )
    output_path = os.path.join(args.output_dir, output_filename)

    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\n✅ All generation results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search over VLM steering configs for generation")

    # Dataset and model
    parser.add_argument("--dataset_name", type=str, default="mmbench", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="llava-llama-3", help="Model from MODEL_NAME_MAP")
    parser.add_argument("--num_samples", type=str, default="all", help="Number of samples to use")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")

    # Paths
    parser.add_argument("--hidden_states_path", type=str, default="data/vlm/hidden_states")
    parser.add_argument("--hidden_states_neg_path", type=str, default="data/vlm/hidden_states_neg")
    parser.add_argument("--output_dir", type=str, default="results/vlm/generation_steering")
    parser.add_argument("--steering_vectors_dir", type=str, default="data/vlm/steering_vectors")

    # Attribution parameters
    parser.add_argument("--attribution", type=str, default="contrastive", choices=["normal", "contrastive"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--grad_method", type=str, default="integrated_gradients")

    # Steering vector search
    parser.add_argument("--mode", nargs="+", default=["pos", "both"])
    parser.add_argument("--methods", nargs="+", default=["pca", "mean"])
    parser.add_argument("--layer_idxs", nargs="+", type=int, default=[28, 29, 30, 31])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])

    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()
    main(args)