from torch.nn.functional import log_softmax

from .utils import (
    compute_signed_attributions,
    apply_smoothgrad,
    apply_smoothgrad_contrastive,
    apply_integrated_gradients,
    apply_integrated_gradients_contrastive,
)


def get_token_attributions(model, processor, image, prompt, response, method="vanilla"):
    """
    Compute token-level attribution scores for a prompt-response pair with image input.
    """
    model.eval()
    model.zero_grad()

    full_prompt = prompt + " " + response
    inputs = processor(images=image, text=full_prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    target_token_id = input_ids[0, -1].item()

    embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

    if method == "vanilla":
        logits = model(inputs_embeds=embeddings, attention_mask=attention_mask).logits[0, -1]
        log_prob = log_softmax(logits, dim=-1)[target_token_id]
        log_prob.backward()
        grads = embeddings.grad[0]

    elif method == "smoothgrad":
        grads = apply_smoothgrad(model, embeddings, attention_mask, target_token_id)[0]

    elif method == "integrated_gradients":
        grads = apply_integrated_gradients(model, embeddings, attention_mask, target_token_id)[0]

    else:
        raise ValueError(f"Unknown attribution method: {method}")

    scores = compute_signed_attributions(grads, embeddings[0])
    return {"attributions": scores, "input_ids": input_ids}


def get_token_attributions_contrastive(model, processor, image, prompt, pos_response, neg_response, method="vanilla"):
    """
    Compute contrastive token-level attributions between positive and negative responses.
    """
    model.eval()
    model.zero_grad()

    pos_input = processor(images=image, text=prompt + " " + pos_response, return_tensors="pt").to(model.device)
    neg_input = processor(images=image, text=prompt + " " + neg_response, return_tensors="pt").to(model.device)

    pos_ids, neg_ids = pos_input["input_ids"], neg_input["input_ids"]
    pos_mask, neg_mask = pos_input["attention_mask"], neg_input["attention_mask"]

    embed_fn = model.get_input_embeddings()
    pos_embed = embed_fn(pos_ids).detach().requires_grad_(True)
    neg_embed = embed_fn(neg_ids).detach().requires_grad_(True)

    pos_target = pos_ids[0, -1].item()
    neg_target = neg_ids[0, -1].item()

    if method == "vanilla":
        pos_logits = model(inputs_embeds=pos_embed, attention_mask=pos_mask).logits[0, -1]
        neg_logits = model(inputs_embeds=neg_embed, attention_mask=neg_mask).logits[0, -1]

        loss = log_softmax(pos_logits, dim=-1)[pos_target] - log_softmax(neg_logits, dim=-1)[neg_target]
        loss.backward()

        pos_scores = compute_signed_attributions(pos_embed.grad[0], pos_embed[0])
        neg_scores = compute_signed_attributions(neg_embed.grad[0], neg_embed[0])

    elif method == "smoothgrad":
        pos_grad, neg_grad = apply_smoothgrad_contrastive(
            model, pos_embed, neg_embed, pos_mask, neg_mask, pos_target, neg_target
        )
        pos_scores = pos_grad.sum(dim=-1).detach().cpu().numpy()
        neg_scores = neg_grad.sum(dim=-1).detach().cpu().numpy()

    elif method == "integrated_gradients":
        pos_grad, neg_grad = apply_integrated_gradients_contrastive(
            model, pos_embed, neg_embed, pos_mask, neg_mask, pos_target, neg_target
        )
        pos_scores = pos_grad.sum(dim=-1).detach().cpu().numpy()
        neg_scores = neg_grad.sum(dim=-1).detach().cpu().numpy()

    else:
        raise ValueError(f"Unknown attribution method: {method}")

    return {
        "pos": (pos_scores, pos_ids),
        "neg": (neg_scores, neg_ids)
    }


def ablate(processor, ids, indices):
    """
    Replace selected token indices with PAD or UNK for ablation.
    """
    ablated = ids.clone()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.unk_token_id
    for i in indices:
        ablated[0, i] = pad_id
    return ablated