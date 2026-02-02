"""GRPO training loop and hyperparameters."""

import torch
import os
import json
import random
import numpy as np
import sys
import wandb

from typing import Literal
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from .drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer
from .utils import (
    init_vllm,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
    get_response_log_probs
)
from .rl_utils import (
    grpo_microbatch_train_step,
    compute_group_normalized_rewards,
    masked_mean,
)

def print_flush(*args, **kwargs):
    """Print with immediate flush for subprocess compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Paths
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/train.jsonl"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "r1_zero.prompt").read_text()

def load_data(data_path: str):
    """Load data."""
    prompts = []
    ground_truths = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Support both "question" and "problem" keys
            question = data.get("question") or data.get("problem")
            prompt = PROMPT_TEMPLATE.format(question=question)
            prompts.append(prompt)
            ground_truths.append(extract_boxed_answer(data["solution"]))
    return prompts, ground_truths

def evaluate(llm, sampling_params, val_prompts: list[str], val_gt_answers: list[str]) -> float:
    """Evaluate model using vLLM and return accuracy."""
    outputs = llm.generate(val_prompts, sampling_params)    
    correct = 0
    for output, ground_truth in zip(outputs, val_gt_answers):
        response = output.outputs[0].text
        reward_info = r1_zero_reward_fn(response, ground_truth)
        reward = reward_info["reward"]
        if reward == 1.0:
            correct += 1
    accuracy = correct / len(val_prompts) * 100
    return accuracy

def train(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    vllm_gpu: int = 0,
    vllm_gpu_memory_utilization: float = 0.35,
    # Define hyperparameters
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 2048,
    epochs_per_rollout_batch: int = 1,  # 1 = on-policy, >1 = off-policy
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,  # microbatch size is 2, will fit on H100
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    max_grad_norm: float = 1.0,  # Default gradient clipping
    cliprange: float = 0.2,  # For GRPO-Clip
    eval_interval: int = 10,  # Evaluate every N steps
    use_wandb: bool = True,
):
    """Run RL training with vLLM for evaluation.
    Device Strategy:
    - If CUDA_VISIBLE_DEVICES is set externally, use the first visible GPU (cuda:0)
    - Otherwise, set CUDA_VISIBLE_DEVICES to vllm_gpu
    - vLLM and training model share the same GPU
    - vllm_gpu_memory_utilization controls vLLM's memory, leaving rest for training
    """
    # Sanity check asserts and derived constants
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    # Check if CUDA_VISIBLE_DEVICES is already set externally
    external_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if external_cuda_visible:
        # Use externally set GPU (will be cuda:0)
        print(f"Using externally set CUDA_VISIBLE_DEVICES={external_cuda_visible}")
        physical_gpu = external_cuda_visible.split(",")[0]
    else:
        # Set CUDA_VISIBLE_DEVICES to the specified vllm_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu)
        physical_gpu = str(vllm_gpu)
    
    # Enable insecure serialization for vLLM apply_model to work with closures
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    print_flush(f"Device configuration:")
    print_flush(f"  Physical GPU: {physical_gpu}")
    print_flush(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print_flush(f"  vLLM memory utilization: {vllm_gpu_memory_utilization}")
    
    # Sampling parameters for vLLM generation
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # Initialize vLLM - it will only see GPU 0 (the physical GPU)
    print_flush(f"\nInitializing vLLM on cuda:0 (physical GPU {physical_gpu})...")
    llm = init_vllm(
        model_id=model_path,
        device="cuda:0",
        seed=42,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    print_flush("vLLM initialized successfully.")
    # Check GPU count - should only see 1 GPU at this point
    print_flush(f"\nPyTorch sees {torch.cuda.device_count()} GPU(s)")
    # Now we need to load the training model on a DIFFERENT physical GPU.
    # Since we set CUDA_VISIBLE_DEVICES=vllm_gpu, torch can only see that GPU.
    # We CANNOT change CUDA_VISIBLE_DEVICES after torch is initialized.
    # 
    # Solution: Load training model on the SAME logical device (cuda:0),
    # but use a LOWER gpu_memory_utilization for vLLM to leave room.
    #
    # However, this is not ideal. The better solution is to run vLLM
    # as a separate server process. For now, we'll share the GPU.
    # Check memory on the single visible GPU
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print_flush(f"\nGPU memory after vLLM init (physical GPU {physical_gpu}):")
    print_flush(f"  cuda:0: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    print_flush(f"\nLoading training model on cuda:0 (physical GPU {physical_gpu})...")
    print_flush("(Training shares GPU with vLLM; vLLM's gpu_memory_utilization limits its memory)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable()
    print_flush("Gradient checkpointing enabled to reduce memory usage")
    
    model = model.to("cuda:0")
    print_flush("Training model loaded successfully.")
    
    # Check GPU memory after training model load
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print_flush(f"\nGPU memory after training model load:")
    print_flush(f"  cuda:0: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    # Load data
    train_prompts, train_gt_answers = load_data(train_data_path)
    print_flush(f"Loaded {len(train_prompts)} training examples")
    val_prompts, val_gt_answers = load_data(val_data_path)
    print_flush(f"Loaded {len(val_prompts)} validation examples")
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    # Initialize wandb
    if use_wandb:
        is_off_policy = epochs_per_rollout_batch > 1
        wandb.init(
            project="grpo-training",
            config={
                "n_grpo_steps": n_grpo_steps,
                "learning_rate": learning_rate,
                "rollout_batch_size": rollout_batch_size,
                "group_size": group_size,
                "epochs_per_rollout_batch": epochs_per_rollout_batch,
                "train_batch_size": train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "micro_train_batch_size": micro_train_batch_size,
                "loss_type": loss_type,
                "use_std_normalization": use_std_normalization,
                "max_grad_norm": max_grad_norm,
                "cliprange": cliprange,
                "is_off_policy": is_off_policy,
            }
        )
    
    # Evaluation sampling params (greedy)
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    global_step = 0
    optimizer.zero_grad()
    
    # Training loop - iterate for n_grpo_steps gradient updates
    data_idx = 0  # Track position in training data
    pbar = tqdm(total=n_grpo_steps, desc="GRPO Training")
    
    while global_step < n_grpo_steps:
        # Sample a batch of prompts (cycling through data)
        batch_prompts = []
        batch_gt_answers = []
        for _ in range(n_prompts_per_rollout_batch):
            batch_prompts.append(train_prompts[data_idx % len(train_prompts)])
            batch_gt_answers.append(train_gt_answers[data_idx % len(train_gt_answers)])
            data_idx += 1
        
        # === Step 1: Generate rollouts with current policy ===
        model.eval()
        load_policy_into_vllm_instance(model, llm)
        outputs = llm.generate(batch_prompts, sampling_params)
        
        batch_responses = []
        for output in outputs:
            for response in output.outputs:
                batch_responses.append(response.text)
        
        # Repeat prompts and ground truths to match rollout_batch_size
        repeated_batch_prompts = []
        repeated_batch_gt_answers = []
        for prompt, gt_answer in zip(batch_prompts, batch_gt_answers):
            repeated_batch_prompts.extend([prompt] * group_size)
            repeated_batch_gt_answers.extend([gt_answer] * group_size)
        
        # === Step 2: Compute rewards and advantages ===
        # Return order: (advantages/normalized_rewards, raw_rewards, metadata)
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn, batch_responses, repeated_batch_gt_answers, 
            group_size, advantage_eps, use_std_normalization
        )
        # Reshape to (batch_size, 1) for broadcasting
        raw_rewards = raw_rewards.unsqueeze(-1).to("cuda:0")
        advantages = advantages.unsqueeze(-1).to("cuda:0")
        
        # === Step 3: Tokenize rollouts ===
        items = tokenize_prompt_and_output(repeated_batch_prompts, batch_responses, tokenizer)
        input_ids = items["input_ids"].to("cuda:0")
        labels = items["labels"].to("cuda:0")
        response_mask = items["response_mask"].to("cuda:0")
        
        # === Step 4: For off-policy, compute old log probs ONCE per rollout batch ===
        # This is reused across all epochs_per_rollout_batch
        if epochs_per_rollout_batch > 1 or loss_type == "grpo_clip":
            with torch.no_grad():
                old_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
        else:
            old_log_probs = None
        
        # === Step 5: Training epochs on this rollout batch ===
        for epoch in range(epochs_per_rollout_batch):
            model.train()
            
            # Accumulators for logging
            epoch_losses = []
            epoch_entropies = []
            epoch_clip_fractions = []
            
            # Microbatch loop
            for j in range(0, rollout_batch_size, micro_train_batch_size):
                # Slice microbatch
                mb_input_ids = input_ids[j:j+micro_train_batch_size]
                mb_labels = labels[j:j+micro_train_batch_size]
                mb_response_mask = response_mask[j:j+micro_train_batch_size]
                mb_raw_rewards = raw_rewards[j:j+micro_train_batch_size]
                mb_advantages = advantages[j:j+micro_train_batch_size]
                mb_old_log_probs = old_log_probs[j:j+micro_train_batch_size] if old_log_probs is not None else None
                
                # Compute policy log probs (with gradients) and entropy
                log_prob_result = get_response_log_probs(
                    model, mb_input_ids, mb_labels, return_token_entropy=True
                )
                policy_log_probs = log_prob_result["log_probs"]
                token_entropy = log_prob_result["token_entropy"]
                
                # Compute loss and backward
                loss, loss_metadata = grpo_microbatch_train_step(
                    policy_log_probs, 
                    mb_response_mask, 
                    gradient_accumulation_steps, 
                    loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                )
                
                # Accumulate for logging
                epoch_losses.append(loss.item() * gradient_accumulation_steps)  # Undo the scaling
                
                # Compute masked mean entropy
                with torch.no_grad():
                    avg_entropy = masked_mean(token_entropy, mb_response_mask).item()
                    epoch_entropies.append(avg_entropy)
                
                # Compute clip fraction if using grpo_clip
                if loss_type == "grpo_clip" and "importance_ratios" in loss_metadata:
                    with torch.no_grad():
                        importance_ratios = loss_metadata["importance_ratios"]
                        # Clipped if ratio is outside [1-cliprange, 1+cliprange]
                        clipped = (importance_ratios < 1 - cliprange) | (importance_ratios > 1 + cliprange)
                        clip_frac = masked_mean(clipped.float(), mb_response_mask).item()
                        epoch_clip_fractions.append(clip_frac)
            
            # Compute gradient norm before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            
            # Gradient clipping
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)
            
            # === Logging ===
            log_dict = {
                "step": global_step,
                "loss": np.mean(epoch_losses),
                "grad_norm": grad_norm,
                "token_entropy": np.mean(epoch_entropies)
            }
            if epoch_clip_fractions:
                log_dict["clip_fraction"] = np.mean(epoch_clip_fractions)
            
            # Periodic evaluation
            if global_step % eval_interval == 0:
                model.eval()
                load_policy_into_vllm_instance(model, llm)
                val_accuracy = evaluate(llm, eval_sampling_params, val_prompts, val_gt_answers)
                log_dict["val/accuracy"] = val_accuracy
                print_flush(f"\nStep {global_step}: Val Accuracy = {val_accuracy:.2f}%")
                model.train()
            
            if use_wandb:
                wandb.log(log_dict)
            
            pbar.set_postfix({
                "loss": f"{np.mean(epoch_losses):.4f}"
            })
            
            if global_step >= n_grpo_steps:
                break
    
    pbar.close()

    # Final evaluation
    print_flush("\nFinal evaluation...")
    model.eval()
    load_policy_into_vllm_instance(model, llm)
    final_accuracy = evaluate(llm, eval_sampling_params, val_prompts, val_gt_answers)
    print_flush(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    
    if use_wandb:
        wandb.log({"final_val_accuracy": final_accuracy})
        wandb.finish()
    
    return model, final_accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--val_data_path", type=str, default=VAL_DATA_PATH)
    parser.add_argument("--vllm_gpu", type=int, default=0)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=2048)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1,
                        help="1 = on-policy, >1 = off-policy")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--use_std_normalization", type=bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    model, final_accuracy = train(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        vllm_gpu=args.vllm_gpu,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        max_grad_norm=args.max_grad_norm,
        cliprange=args.cliprange,
        eval_interval=args.eval_interval,
        use_wandb=not args.no_wandb,
    )
    print_flush(f"Final accuracy: {final_accuracy}")

if __name__ == "__main__":
    main()