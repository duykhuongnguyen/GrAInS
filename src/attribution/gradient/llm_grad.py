from torch.nn.functional import log_softmax

from .utils import (
    compute_signed_attributions,
    apply_smoothgrad,
    apply_smoothgrad_contrastive,
    apply_integrated_gradients,
    apply_integrated_gradients_contrastive,
)


def get_token_attributions(model, tokenizer, prompt, response, method="vanilla"):
    """
    Compute token-level attribution scores for the prompt (excluding response).
    """
    model.eval()
    model.zero_grad()

    full_input = tokenizer(prompt + " " + response, return_tensors="pt").to(model.device)
    input_ids = full_input["input_ids"]
    attention_mask = full_input["attention_mask"]
    target_token_id = input_ids[0, -1].item()

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach().requires_grad_(True)

    if method == "vanilla":
        logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits[0, -1]
        log_prob = log_softmax(logits, dim=-1)[target_token_id]
        log_prob.backward()
        grads = inputs_embeds.grad[0]

    elif method == "smoothgrad":
        grads = apply_smoothgrad(model, inputs_embeds, attention_mask, target_token_id)[0]

    elif method == "integrated_gradients":
        grads = apply_integrated_gradients(model, inputs_embeds, attention_mask, target_token_id)[0]

    else:
        raise ValueError(f"Unknown attribution method: {method}")

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(input_ids.device)
    prompt_len = prompt_ids.shape[0]

    grads = grads[:prompt_len]
    inputs_embeds_prompt = inputs_embeds[0][:prompt_len]
    token_strs = tokenizer.convert_ids_to_tokens(input_ids[0][:prompt_len])
    signed_scores = compute_signed_attributions(grads, inputs_embeds_prompt)

    return token_strs, signed_scores, input_ids[:, :prompt_len]


def get_token_attributions_contrastive(model, tokenizer, prompt, pos_response, neg_response, method="vanilla"):
    """
    Compute contrastive token attributions between positive and negative responses.
    """
    model.eval()
    model.zero_grad()

    pos_input = tokenizer(prompt + " " + pos_response, return_tensors="pt").to(model.device)
    neg_input = tokenizer(prompt + " " + neg_response, return_tensors="pt").to(model.device)

    pos_ids, pos_mask = pos_input["input_ids"], pos_input["attention_mask"]
    neg_ids, neg_mask = neg_input["input_ids"], neg_input["attention_mask"]

    embedding_layer = model.get_input_embeddings()
    pos_embeds = embedding_layer(pos_ids).detach().requires_grad_(True)
    neg_embeds = embedding_layer(neg_ids).detach().requires_grad_(True)

    pos_target = pos_ids[0, -1].item()
    neg_target = neg_ids[0, -1].item()

    if method == "vanilla":
        pos_logits = model(inputs_embeds=pos_embeds, attention_mask=pos_mask).logits[0, -1]
        neg_logits = model(inputs_embeds=neg_embeds, attention_mask=neg_mask).logits[0, -1]

        loss = log_softmax(pos_logits, dim=-1)[pos_target] - log_softmax(neg_logits, dim=-1)[neg_target]
        loss.backward()

        pos_scores = compute_signed_attributions(pos_embeds.grad[0], pos_embeds[0])
        neg_scores = compute_signed_attributions(neg_embeds.grad[0], neg_embeds[0])

    elif method == "smoothgrad":
        pos_grad, neg_grad = apply_smoothgrad_contrastive(
            model, pos_embeds, neg_embeds, pos_mask, neg_mask, pos_target, neg_target
        )
        pos_scores = pos_grad.sum(dim=-1).detach().cpu().numpy()
        neg_scores = neg_grad.sum(dim=-1).detach().cpu().numpy()

    elif method == "integrated_gradients":
        pos_grad, neg_grad = apply_integrated_gradients_contrastive(
            model, pos_embeds, neg_embeds, pos_mask, neg_mask, pos_target, neg_target
        )
        pos_scores = pos_grad.sum(dim=-1).detach().cpu().numpy()
        neg_scores = neg_grad.sum(dim=-1).detach().cpu().numpy()

    else:
        raise ValueError(f"Unknown attribution method: {method}")

    return {
        "pos": (pos_scores, pos_ids),
        "neg": (neg_scores, neg_ids),
    }


def ablate(tokenizer, ids, indices):
    """
    Replace selected token indices with PAD or UNK for ablation.
    """
    ablated = ids.clone()
    pad_or_unk = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id
    for i in indices:
        ablated[0, i] = pad_or_unk
    return ablated