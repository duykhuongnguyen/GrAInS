import os
import argparse
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data import load_data
from utils.model import load_llm_model_and_tokenizer
from utils.steering import (
    load_hidden_states,
    compute_steering_vector,
    generate_answer_dataset,
    steer_and_generate_dataset,
)
from utils.config import MODEL_NAME_MAP


def run_single_eval(args, method, layer_idx, alpha, mode, base_outputs):
    """
    Evaluate generation with a single steering vector configuration.
    """
    # Compute steering vector
    hidden_states = load_hidden_states(args.hidden_states_path)
    steering_vec = compute_steering_vector(hidden_states, layer_idx, method)

    if mode == "both":
        hidden_states_neg = load_hidden_states(args.hidden_states_neg_path)
        steering_vec_neg = compute_steering_vector(hidden_states_neg, layer_idx, method)
        steering_vec -= steering_vec_neg

    # Save steering vector
    os.makedirs(args.steering_vectors_dir, exist_ok=True)
    vec_filename = (
        f"steering_{args.dataset_name}_{args.model_name}_{method}_"
        f"layer{layer_idx}_alpha{alpha}_{mode}.npy"
    )
    vec_path = os.path.join(args.steering_vectors_dir, vec_filename)
    np.save(vec_path, steering_vec)

    # Generate outputs with steering
    steered_outputs = steer_and_generate_dataset(
        dataset=args.dataset,
        model=args.model,
        tokenizer=args.tokenizer,
        steering_vec=steering_vec,
        layer_idx=layer_idx,
        alpha=alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    results = []
    for example, steered_output in zip(args.dataset, steered_outputs):
        question = example["question"].strip()
        choices = example["choices"]
        label = example["label"]
        prompt = f"Q: {question}\nA:"
        true_answer = choices[label].strip()
        base_output = base_outputs[prompt]

        results.append({
            "prompt": prompt,
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
    model, tokenizer = load_llm_model_and_tokenizer(MODEL_NAME_MAP[args.model_name])
    dataset = load_data(args.dataset_name, args.num_samples, args.split)

    args.model = model
    args.tokenizer = tokenizer
    args.dataset = dataset

    # Generate base (non-steered) outputs
    print("Generating base outputs...")
    base_generations = generate_answer_dataset(
        dataset, model, tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    base_outputs = {
        f"Q: {ex['question'].strip()}\nA:": output
        for ex, output in zip(dataset, base_generations)
    }

    all_results = []

    print("Running generation with steering vectors...")
    search_space = product(args.methods, args.layer_idxs, args.alphas, args.mode)
    total_combinations = len(args.methods) * len(args.layer_idxs) * len(args.alphas) * len(args.mode)

    for method, layer_idx, alpha, mode in tqdm(search_space, total=total_combinations):
        results = run_single_eval(args, method, layer_idx, alpha, mode, base_outputs)
        all_results.extend(results)

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = (
        f"generation_results_{args.dataset_name}_{args.model_name}_"
        f"{args.num_samples}_{args.attribution}_{args.top_k}_{args.grad_method}.csv"
    )
    output_path = os.path.join(args.output_dir, output_filename)

    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\nAll generation results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM generation")

    # Dataset and model
    parser.add_argument("--dataset_name", type=str, default="truthfulqa", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b", help="Model name from MODEL_NAME_MAP")
    parser.add_argument("--num_samples", type=str, default="all", help="Number of samples to use")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")

    # Paths
    parser.add_argument("--hidden_states_path", type=str, default="data/llm/hidden_states", help="Path to positive hidden states")
    parser.add_argument("--hidden_states_neg_path", type=str, default="data/llm/hidden_states_neg", help="Path to negative hidden states")
    parser.add_argument("--output_dir", type=str, default="results/llm/generation_steering", help="Directory to save output CSV")
    parser.add_argument("--steering_vectors_dir", type=str, default="data/llm/steering_vectors", help="Directory to save steering vectors")

    # Attribution parameters
    parser.add_argument("--attribution", type=str, default="contrastive", choices=["normal", "contrastive"], help="Attribution mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top-k tokens to ablate")
    parser.add_argument("--grad_method", type=str, default="integrated_gradients", help="Attribution method")

    # Steering vector parameters
    parser.add_argument("--mode", nargs="+", default=["pos", "both"], help="Modes: ['pos', 'both']")
    parser.add_argument("--methods", nargs="+", default=["pca", "mean"], help="Steering vector methods")
    parser.add_argument("--layer_idxs", nargs="+", type=int, default=[28, 29, 30, 31], help="Transformer layer indices")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="Scaling factors")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")

    args = parser.parse_args()
    main(args)