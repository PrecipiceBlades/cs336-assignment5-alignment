"""Utility functions for reinforcement learning (GRPO)."""

from typing import Callable, Literal
from einops import rearrange

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: Number of responses per question (group).
        advantage_eps: Small constant to avoid division by zero in normalization.
        normalize_by_std: If True, divide by the per-group standard deviation;
            otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages: shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata: your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    # TODO: Implement this function
    reward_infos = [reward_fn(response, gt) for response, gt in zip(rollout_responses, repeated_ground_truths)]
    rewards = torch.tensor([info["reward"] for info in reward_infos])
    format_rewards = torch.tensor([info.get("format_reward", 0.0) for info in reward_infos])
    answer_rewards = torch.tensor([info.get("answer_reward", 0.0) for info in reward_infos])
    
    rewards_grouped = rearrange(rewards, "(n g) -> n g", g=group_size)
    mean = torch.mean(rewards_grouped, dim=1, keepdim=True)
    std = torch.std(rewards_grouped, dim=1, keepdim=True)
    if normalize_by_std:
        advantages = (rewards_grouped - mean) / (std + advantage_eps)
    else:
        advantages = rewards_grouped - mean
    
    # Return order: (advantages/normalized_rewards, raw_rewards, metadata)
    return advantages.flatten(), rewards.flatten(), {
        "mean": mean,
        "std": std,
        "reward_mean": rewards.mean().item(),
        "format_reward_mean": format_rewards.mean().item(),
        "answer_reward_mean": answer_rewards.mean().item(),
    }


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the policy-gradient loss at every token.

    The naive per-token policy gradient loss is: -A_t * log p_theta(o_t | q, o_{<t})

    Args:
        raw_rewards_or_advantages: Shape (batch_size, 1), scalar reward/advantage
            for each rollout response.
        policy_log_probs: Shape (batch_size, sequence_length), logprobs for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in the
            training loop).

    Implementation tips:
        - Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
    """
    # TODO: Implement this function
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the per-token GRPO-Clip loss.

    The per-token GRPO-Clip loss is:
        -min( (pi_theta / pi_theta_old) * A_t,
              clip(pi_theta / pi_theta_old, 1 - epsilon, 1 + epsilon) * A_t )

    Args:
        advantages: Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: Shape (batch_size, sequence_length), per-token log probs
            from the policy being trained.
        old_log_probs: Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: Clip parameter epsilon (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor of shape (batch_size, sequence_length), the per-token
                clipped loss.
            metadata: dict containing whatever you want to log. We suggest logging
                whether each token was clipped or not, i.e., whether the clipped
                policy gradient loss on the RHS of the min was lower than the LHS.

    Implementation tips:
        - Broadcast advantages over sequence_length.
    """
    # TODO: Implement this function
    importance_ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_importance_ratios = torch.clamp(importance_ratios, 1 - cliprange, 1 + cliprange)
    loss = -torch.min(importance_ratios * advantages, clipped_importance_ratios * advantages)
    return loss, {
        "importance_ratios": importance_ratios,
        "clipped_importance_ratios": clipped_importance_ratios,
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select and compute the desired policy-gradient loss.

    A convenience wrapper that dispatches to the correct loss routine
    (no_baseline, reinforce_with_baseline, or grpo_clip) and returns both
    the per-token loss and any auxiliary statistics.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities
            from the policy being trained.
        loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Required for "reinforce_with_baseline" and "grpo_clip";
            shape (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"; scalar epsilon used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: (batch_size, sequence_length), per-token loss.
            metadata: dict, statistics from the underlying routine (e.g., clip
                fraction for GRPO-Clip).

    Implementation tips:
        - Delegate to compute_naive_policy_gradient_loss or compute_grpo_clip_loss.
        - Perform argument checks (see assertion pattern above).
        - Aggregate any returned metadata into a single dict.
    """
    # TODO: Implement this function
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of tensor along a given dimension, considering only masked elements.

    Averages tensor elements while respecting a boolean mask.

    Args:
        tensor: The data to be averaged.
        mask: Same shape as tensor; positions with 1 are included in the mean.
        dim: Dimension over which to average. If None, compute the mean over all
            masked elements.

    Returns:
        torch.Tensor: The masked mean; shape matches tensor.mean(dim) semantics.
    """
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    constant_normalizer: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only masked elements.

    Unlike masked_mean which divides by the actual count of masked elements,
    this divides by a fixed constant (e.g., max sequence length).

    Args:
        tensor: The data to sum and normalize.
        mask: Same shape as tensor; positions with 1 are included in the sum.
        dim: Dimension over which to sum before normalization. If None, sum over
            all dimensions.
        constant_normalizer: The constant to divide by for normalization.

    Returns:
        torch.Tensor: The normalized sum, where masked elements (mask=0) don't contribute.
    """
    return torch.sum(tensor * mask, dim=dim) / constant_normalizer


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.

    Implement a single micro-batch update for GRPO, including policy-gradient loss,
    averaging with a mask, and gradient scaling.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities
            from the policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs: Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange: Clip parameter epsilon for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation.
                We return this so we can log it.
            metadata: Dict with metadata from the underlying loss call, and any other
                statistics you might want to log.

    Implementation tips:
        - You should call loss.backward() in this function. Make sure to adjust for
          gradient accumulation.
    """
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, metadata

def dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str) -> torch.Tensor:
    """Compute the DPO loss for a single example.

    The DPO loss is:
        -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    where log_ratio = log pi(y|x) - log pi_ref(y|x)

    Args:
        lm: The model to train.
        lm_ref: The reference model.
        tokenizer: The tokenizer to use.
        beta: The DPO beta hyperparameter.
        prompt: The prompt to generate responses for.
        response_chosen: The chosen response.
        response_rejected: The rejected response.
    
    Returns:
        torch.Tensor: Scalar DPO loss for this example.
    """
    from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs
    
    device = next(lm.parameters()).device
    
    # Tokenize prompt + chosen response
    chosen_batch = tokenize_prompt_and_output([prompt], [response_chosen], tokenizer)
    chosen_input_ids = chosen_batch["input_ids"].to(device)
    chosen_labels = chosen_batch["labels"].to(device)
    chosen_mask = chosen_batch["response_mask"].to(device)
    
    # Tokenize prompt + rejected response
    rejected_batch = tokenize_prompt_and_output([prompt], [response_rejected], tokenizer)
    rejected_input_ids = rejected_batch["input_ids"].to(device)
    rejected_labels = rejected_batch["labels"].to(device)
    rejected_mask = rejected_batch["response_mask"].to(device)
    
    # Get log probs from policy model
    policy_chosen_log_probs = get_response_log_probs(lm, chosen_input_ids, chosen_labels)["log_probs"]
    policy_rejected_log_probs = get_response_log_probs(lm, rejected_input_ids, rejected_labels)["log_probs"]
    
    # Get log probs from reference model (no gradients needed)
    with torch.no_grad():
        ref_chosen_log_probs = get_response_log_probs(lm_ref, chosen_input_ids, chosen_labels)["log_probs"]
        ref_rejected_log_probs = get_response_log_probs(lm_ref, rejected_input_ids, rejected_labels)["log_probs"]
    
    # Sum log probs over response tokens only (using mask)
    policy_chosen_sum = (policy_chosen_log_probs * chosen_mask).sum()
    policy_rejected_sum = (policy_rejected_log_probs * rejected_mask).sum()
    ref_chosen_sum = (ref_chosen_log_probs * chosen_mask).sum()
    ref_rejected_sum = (ref_rejected_log_probs * rejected_mask).sum()
    
    # Compute log ratios: log pi(y|x) - log pi_ref(y|x)
    log_ratio_chosen = policy_chosen_sum - ref_chosen_sum
    log_ratio_rejected = policy_rejected_sum - ref_rejected_sum
    
    # DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    loss = -torch.nn.functional.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    
    return loss
