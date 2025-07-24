import torch
from torch.nn.functional import log_softmax
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForVision2Seq,
)
from accelerate import Accelerator


def load_llm_model_and_tokenizer(model_name):
    """
    Load a language model and tokenizer with accelerator support.
    """
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model, tokenizer = accelerator.prepare(model, tokenizer)
    return model, tokenizer


def load_vlm_model_and_processor(model_name):
    """
    Load a vision-language model and processor with accelerator support.
    """
    accelerator = Accelerator()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    model, processor = accelerator.prepare(model, processor)
    return model, processor


def get_all_layer_hidden_states(model, input_ids):
    """
    Get hidden states from all layers for the last token in the input sequence.
    Returns:
        np.ndarray: Array of shape (num_layers, hidden_dim).
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states
    return torch.stack([layer[0, -1] for layer in hidden_states], dim=0).cpu().numpy()


def get_all_layer_hidden_states_all_tokens(model, input_ids):
    """
    Get hidden states from all layers for all tokens in the input sequence.
    Returns:
        np.ndarray: Array of shape (num_layers, seq_len, hidden_dim).
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states
    return torch.stack([layer[0] for layer in hidden_states], dim=0).cpu().numpy()


def get_log_prob(model, input_ids):
    """
    Compute the log-probability of the last token in the input sequence.
    Returns:
        float: Log-probability of the last token.
    """
    logits = model(input_ids=input_ids).logits[0, -1]
    log_probs = log_softmax(logits, dim=-1)
    return log_probs[input_ids[0, -1].item()].item()


def get_top_token_indices(scores, input_ids, top_k, top):
    """Select top-k token indices based on attribution scores and strategy."""
    if top == "pos":
        return list(np.argsort(scores)[::-1])[:top_k]
    elif top == "abs":
        return list(np.argsort(np.abs(scores))[::-1])[:top_k]
    elif top == "neg":
        return list(np.argsort(scores))[:top_k]
    elif top == "rand":
        return list(np.random.choice(len(input_ids[0]), size=min(top_k, len(input_ids[0])), replace=False))
    else:
        raise ValueError(f"Invalid 'top' value: {top}. Use 'pos', 'abs', 'neg', or 'rand'.")