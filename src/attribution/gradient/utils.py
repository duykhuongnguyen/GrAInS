import torch
from torch.nn.functional import log_softmax
from captum.attr import IntegratedGradients


def compute_signed_attributions(grads, embeddings):
    """
    Computes signed attribution scores via the dot product of gradients and embeddings.
    """
    return torch.einsum("ij,ij->i", grads, embeddings).detach().cpu().numpy()


def apply_smoothgrad(model, inputs_embeds, attention_mask, target_token_id, n_samples=20, noise_std=0.01):
    """
    Applies SmoothGrad to compute token-level attributions.

    Args:
        model: Language model
        inputs_embeds: Token embeddings
        attention_mask: Attention mask
        target_token_id: ID of the target token for attribution
        n_samples: Number of noisy samples
        noise_std: Standard deviation of noise to add

    Returns:
        Averaged gradients across noisy samples
    """
    accumulated_grads = torch.zeros_like(inputs_embeds)

    for _ in range(n_samples):
        noisy_embeds = inputs_embeds + torch.randn_like(inputs_embeds) * noise_std
        noisy_embeds.requires_grad_(True)

        logits = model(inputs_embeds=noisy_embeds, attention_mask=attention_mask).logits[0, -1]
        log_prob = log_softmax(logits, dim=-1)[target_token_id]

        model.zero_grad()
        log_prob.backward()

        if noisy_embeds.grad is not None:
            accumulated_grads += noisy_embeds.grad

    return accumulated_grads / n_samples


def apply_smoothgrad_contrastive(
    model,
    pos_embeds,
    neg_embeds,
    attention_mask_pos,
    attention_mask_neg,
    target_token_id_pos,
    target_token_id_neg,
    n_samples=20,
    noise_std=0.01
):
    """
    Contrastive SmoothGrad between positive and negative embeddings.
    """
    model.eval()
    accumulated_grads_pos = torch.zeros_like(pos_embeds)
    accumulated_grads_neg = torch.zeros_like(neg_embeds)

    with torch.no_grad():
        pos_logits = model(inputs_embeds=pos_embeds, attention_mask=attention_mask_pos).logits[0, -1]
        neg_logits = model(inputs_embeds=neg_embeds, attention_mask=attention_mask_neg).logits[0, -1]

        log_prob_pos = log_softmax(pos_logits, dim=-1)[target_token_id_pos]
        log_prob_neg = log_softmax(neg_logits, dim=-1)[target_token_id_neg]

    # Positive perturbation
    for _ in range(n_samples):
        noisy_pos = (pos_embeds + torch.randn_like(pos_embeds) * noise_std).detach().requires_grad_(True)
        logits = model(inputs_embeds=noisy_pos, attention_mask=attention_mask_pos).logits[0, -1]
        log_prob = log_softmax(logits, dim=-1)[target_token_id_pos]
        loss = log_prob - log_prob_neg

        model.zero_grad()
        loss.backward()
        accumulated_grads_pos += noisy_pos.grad.detach()

    # Negative perturbation
    for _ in range(n_samples):
        noisy_neg = (neg_embeds + torch.randn_like(neg_embeds) * noise_std).detach().requires_grad_(True)
        logits = model(inputs_embeds=noisy_neg, attention_mask=attention_mask_neg).logits[0, -1]
        log_prob = log_softmax(logits, dim=-1)[target_token_id_pos]
        loss = log_prob_pos - log_prob

        model.zero_grad()
        loss.backward()
        accumulated_grads_neg += noisy_neg.grad.detach()

    return accumulated_grads_pos[0] / n_samples, accumulated_grads_neg[0] / n_samples


def apply_integrated_gradients(model, inputs_embeds, attention_mask, target_token_id, steps=20):
    """
    Applies Integrated Gradients (IG) to compute token attributions.

    Args:
        model: Language model
        inputs_embeds: Token embeddings
        attention_mask: Attention mask
        target_token_id: Target token for attribution
        steps: Number of interpolation steps

    Returns:
        IG attribution tensor of shape (seq_len, hidden_dim)
    """
    baseline = torch.zeros_like(inputs_embeds)
    total_grads = torch.zeros_like(inputs_embeds)

    for alpha in torch.linspace(0, 1, steps):
        interpolated = baseline + alpha * (inputs_embeds - baseline)
        interpolated.requires_grad_(True)

        logits = model(inputs_embeds=interpolated, attention_mask=attention_mask).logits[0, -1]
        log_prob = log_softmax(logits, dim=-1)[target_token_id]

        model.zero_grad()
        log_prob.backward()

        if interpolated.grad is not None:
            total_grads += interpolated.grad

    avg_grads = total_grads / steps
    return (inputs_embeds - baseline) * avg_grads


def apply_integrated_gradients_contrastive(
    model,
    pos_embeds,
    neg_embeds,
    attention_mask_pos,
    attention_mask_neg,
    target_token_id_pos,
    target_token_id_neg,
    steps=20
):
    """
    Applies Integrated Gradients in a contrastive setting between positive and negative prompts.
    """
    model.eval()
    T = pos_embeds.shape[1]

    joint_input = torch.cat([pos_embeds, neg_embeds], dim=1)
    baseline = torch.zeros_like(joint_input)

    def contrastive_fn(joint_embeds):
        pos_embed = joint_embeds[:, :T, :]
        neg_embed = joint_embeds[:, T:, :]

        pos_logits = model(inputs_embeds=pos_embed, attention_mask=attention_mask_pos).logits[:, -1, :]
        neg_logits = model(inputs_embeds=neg_embed, attention_mask=attention_mask_neg).logits[:, -1, :]

        log_p_pos = log_softmax(pos_logits, dim=-1)[:, target_token_id_pos]
        log_p_neg = log_softmax(neg_logits, dim=-1)[:, target_token_id_neg]

        return log_p_pos - log_p_neg

    ig = IntegratedGradients(contrastive_fn)
    attributions = ig.attribute(joint_input, baselines=baseline, n_steps=steps)

    return attributions[:, :T, :][0], attributions[:, T:, :][0]