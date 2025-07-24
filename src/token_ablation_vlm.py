import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import requests

from attribution.gradient import vlm_grad
from utils.data import load_data
from utils.model import (
    load_vlm_model_and_processor,
    get_all_layer_hidden_states,
    get_log_prob,
    get_top_token_indices
)
from utils.config import MODEL_NAME_MAP


def run_token_ablation(model, processor, image, prompt, pos_response, neg_response, top_k=3, top="pos", attribution_method="vanilla"):
    results_pos = vlm_grad.get_token_attributions(model, processor, image, prompt, pos_response, method=attribution_method)
    results_neg = vlm_grad.get_token_attributions(model, processor, image, prompt, neg_response, method=attribution_method)

    scores_pos, input_ids_pos = results_pos["attributions"], results_pos["input_ids"]
    scores_neg, input_ids_neg = results_neg["attributions"], results_neg["input_ids"]

    top_ids_pos = get_top_token_indices(scores_pos, input_ids_pos, top_k, top)
    top_ids_neg = get_top_token_indices(scores_neg, input_ids_neg, top_k, top)

    ablated_pos_input_ids = vlm_grad.ablate(processor, input_ids_pos.clone(), top_ids_pos)
    ablated_neg_input_ids = vlm_grad.ablate(processor, input_ids_neg.clone(), top_ids_neg)

    return {
        "base_delta": get_log_prob(model, input_ids_pos) - get_log_prob(model, input_ids_neg),
        "ablated_delta": get_log_prob(model, ablated_pos_input_ids) - get_log_prob(model, ablated_neg_input_ids),
        "log_prob_pos": get_log_prob(model, input_ids_pos),
        "log_prob_neg": get_log_prob(model, input_ids_neg),
        "log_prob_pos_ablated": get_log_prob(model, ablated_pos_input_ids),
        "log_prob_neg_ablated": get_log_prob(model, ablated_neg_input_ids),
        "hidden_pos_response": get_all_layer_hidden_states(model, input_ids_pos),
        "hidden_neg_response": get_all_layer_hidden_states(model, input_ids_neg),
        "hidden_pos_response_ablated": get_all_layer_hidden_states(model, ablated_pos_input_ids),
        "hidden_neg_response_ablated": get_all_layer_hidden_states(model, ablated_neg_input_ids),
    }


def run_token_ablation_contrastive(model, processor, image, prompt, pos_response, neg_response, top_k=3, top="pos", attribution_method="vanilla"):
    results_dict = vlm_grad.get_token_attributions_contrastive(
        model, processor, image, prompt, pos_response, neg_response, method=attribution_method
    )

    pos_signed_scores, pos_input_ids = results_dict["pos"]
    neg_signed_scores, neg_input_ids = results_dict["neg"]

    top_ids_pos = get_top_token_indices(pos_signed_scores, pos_input_ids, top_k, top)
    top_ids_neg = get_top_token_indices(neg_signed_scores, neg_input_ids, top_k, top)

    ablated_pos_input_ids = vlm_grad.ablate(processor, pos_input_ids.clone(), top_ids_pos)
    ablated_neg_input_ids = vlm_grad.ablate(processor, neg_input_ids.clone(), top_ids_neg)

    return {
        "base_delta": get_log_prob(model, pos_input_ids) - get_log_prob(model, neg_input_ids),
        "ablated_delta": get_log_prob(model, ablated_pos_input_ids) - get_log_prob(model, ablated_neg_input_ids),
        "log_prob_pos": get_log_prob(model, pos_input_ids),
        "log_prob_neg": get_log_prob(model, neg_input_ids),
        "log_prob_pos_ablated": get_log_prob(model, ablated_pos_input_ids),
        "log_prob_neg_ablated": get_log_prob(model, ablated_neg_input_ids),
        "hidden_pos_response": get_all_layer_hidden_states(model, pos_input_ids),
        "hidden_neg_response": get_all_layer_hidden_states(model, neg_input_ids),
        "hidden_pos_response_ablated": get_all_layer_hidden_states(model, ablated_pos_input_ids),
        "hidden_neg_response_ablated": get_all_layer_hidden_states(model, ablated_neg_input_ids),
    }


def main(args):
    model, processor = load_vlm_model_and_processor(MODEL_NAME_MAP[args.model_name])
    dataset = load_data(args.dataset_name, args.num_samples, args.split)

    all_records = []
    hidden_state_batches = {}

    ablation_func = run_token_ablation if args.attribution == "normal" else run_token_ablation_contrastive

    for example in tqdm(dataset):
        question = example["question"].strip()
        image = example["image"]
        pos_response = example["chosen"].strip()
        neg_response = example["rejected"].strip()

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        result = ablation_func(
            model, processor, image, prompt, pos_response, neg_response,
            top_k=args.top_k, top=args.top, attribution_method=args.grad_method
        )

        for k in list(result.keys()):
            if k.startswith("hidden_"):
                hidden_state_batches.setdefault(k, []).append(result.pop(k))

        result.update({
            "prompt": prompt,
            "pos_response": pos_response,
            "neg_response": neg_response,
        })
        all_records.append(result)

    os.makedirs(args.output_path_hidden, exist_ok=True)
    os.makedirs(args.output_path_meta, exist_ok=True)

    # Save hidden states
    hidden_out_path = os.path.join(
        args.output_path_hidden,
        f"hidden_states_{args.dataset_name}_{args.model_name}_{args.num_samples}_{args.attribution}_{args.top_k}_{args.top}_{args.grad_method}.npz"
    )
    np.savez(hidden_out_path, **{k: np.stack(v, axis=0) for k, v in hidden_state_batches.items()})
    print(f"Saved hidden states to: {hidden_out_path}")

    # Save metadata/log-probs
    record_out_path = os.path.join(
        args.output_path_meta,
        f"token_ablation_{args.dataset_name}_{args.model_name}_{args.num_samples}_{args.attribution}_{args.top_k}_{args.top}_{args.grad_method}.csv"
    )
    pd.DataFrame(all_records).to_csv(record_out_path, index=False)
    print(f"Saved metadata/log-probs to: {record_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token ablation attribution experiment")

    # Dataset and model
    parser.add_argument("--dataset_name", type=str, default="spa-vl", help="Dataset name to load")
    parser.add_argument("--model_name", type=str, default="llava-1.6-7b", help="Model name from MODEL_NAME_MAP")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of QA samples to run")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")

    # Attribution parameters
    parser.add_argument("--top_k", type=int, default=5, help="Top-k tokens to ablate")
    parser.add_argument("--top", type=str, default="pos", choices=["pos", "abs", "neg", "rand"], help="Scoring mode for top-k selection")
    parser.add_argument("--attribution", type=str, default="contrastive", choices=["normal", "contrastive"], help="Ablation mode")
    parser.add_argument("--grad_method", type=str, default="vanilla", help="Attribution method to apply")

    # Output paths
    parser.add_argument("--output_path_meta", type=str, default="results/vlm/token_attribution", help="Path to save metadata/log-probs")
    parser.add_argument("--output_path_hidden", type=str, default="data/vlm/hidden_states", help="Path to save stacked hidden states")

    args = parser.parse_args()
    main(args)