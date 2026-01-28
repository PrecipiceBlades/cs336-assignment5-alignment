"""
SFT (Supervised Fine-Tuning) on MATH dataset.

Run SFT on reasoning examples using Qwen 2.5 Math 1.5B base model.
"""
import torch
import json
import wandb

from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import random
import numpy as np
from unittest.mock import patch

from vllm import LLM


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from .utils import (
    init_vllm_multi_gpu,
    init_vllm,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from .drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn

# Paths
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "r1_zero.prompt").read_text()

# Sampling parameters for vLLM generation
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=2048,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)


def setup_wandb_metrics():
    """Setup wandb metrics for train and eval steps."""
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def load_train_data(data_path: str, num_samples: int = None):
    """Load training data."""
    prompts = []
    responses = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            response = data["response"].removeprefix("<think>")
            prompt = PROMPT_TEMPLATE.format(question=question).removesuffix("<think>")
            prompts.append(prompt)
            responses.append(response)
            if num_samples and len(prompts) >= num_samples:
                break
    return prompts, responses


def load_val_data(data_path: str):
    """Load validation data."""
    prompts = []
    ground_truths = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["problem"]
            prompt = PROMPT_TEMPLATE.format(question=question)
            prompts.append(prompt)
            ground_truths.append(extract_boxed_answer(data["solution"]))
    return prompts, ground_truths


def evaluate(llm, val_prompts: list[str], val_ground_truths: list[str]) -> float:
    """Evaluate model using vLLM and return accuracy."""
    outputs = llm.generate(val_prompts, SAMPLING_PARAMS)
    
    correct = 0
    for output, ground_truth in zip(outputs, val_ground_truths):
        response = output.outputs[0].text
        if not response.endswith("</answer>"):
            response = response + "</answer>"
        reward_info = r1_zero_reward_fn(response, ground_truth)
        if reward_info["answer_reward"] == 1.0:
            correct += 1
    
    accuracy = correct / len(val_prompts) * 100
    return accuracy


def train(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    num_samples: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    num_epochs: int,
    eval_every_steps: int,
    train_devices: list[int] = None,
    vllm_tensor_parallel_size: int = 4,
    use_wandb: bool = True,
):
    """Run SFT training with multi-GPU support.
    
    Note: Run with CUDA_VISIBLE_DEVICES=1,0 so that:
    - cuda:0 maps to physical GPU 1 (for vLLM)
    - cuda:1 maps to physical GPU 0 (for training)
    """
    import os
    
    if train_devices is None:
        train_devices = [0, 1, 2, 3]
    
    # vLLM will use cuda:0 (first visible GPU), training uses cuda:1 (second visible GPU)
    vllm_device = "cuda:0"
    train_device = "cuda:1"
    
    # Initialize vLLM FIRST (it uses cuda:0 by default)
    # Using low gpu_memory_utilization due to existing memory usage on GPUs
    print(f"Initializing vLLM on {vllm_device}...")
    llm = init_vllm(
        model_id=model_path,
        device=vllm_device,
        seed=42,
        gpu_memory_utilization=0.08,  # ~6.5GB, should fit in remaining memory
    )
    
    # Load training model on cuda:1
    print(f"Loading training model from {model_path} on {train_device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(train_device)
    
    # Set primary_device for the rest of training
    primary_device = train_device
    
    # Load data
    print(f"Loading training data ({num_samples} samples)...")
    train_prompts, train_responses = load_train_data(train_data_path, num_samples)
    print(f"Loaded {len(train_prompts)} training examples")
    
    print("Loading validation data...")
    val_prompts, val_ground_truths = load_val_data(val_data_path)
    print(f"Loaded {len(val_prompts)} validation examples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    global_step = 0
    model.train()
    
    # Get the underlying model for weight loading (handle DataParallel wrapper)
    def get_base_model(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for i in tqdm(range(0, len(train_prompts), batch_size), desc=f"Epoch {epoch+1}"):
            batch_prompts = train_prompts[i:i+batch_size]
            batch_responses = train_responses[i:i+batch_size]
            
            # Tokenize
            items = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = items["input_ids"].to(primary_device)
            labels = items["labels"].to(primary_device)
            response_mask = items["response_mask"].to(primary_device)
            
            # Forward pass (use base model for get_response_log_probs)
            base_model = get_base_model(model)
            log_probs = get_response_log_probs(base_model, input_ids, labels)["log_probs"]
            
            # Backward pass (scaled for gradient accumulation)
            loss, _ = sft_microbatch_train_step(
                log_probs, response_mask, gradient_accumulation_steps
            )
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Optimizer step
            if (num_batches % gradient_accumulation_steps == 0) or (i + batch_size >= len(train_prompts)):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log training loss
                if use_wandb:
                    wandb.log({
                        "train/loss": epoch_loss / num_batches,
                        "train_step": global_step,
                    })
                
                # Evaluate periodically
                if global_step % eval_every_steps == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    model.eval()
                    load_policy_into_vllm_instance(get_base_model(model), llm)
                    accuracy = evaluate(llm, val_prompts, val_ground_truths)
                    print(f"Step {global_step}: Validation Accuracy = {accuracy:.2f}%")
                    
                    if use_wandb:
                        wandb.log({
                            "eval/accuracy": accuracy,
                            "eval_step": global_step,
                        })
                    model.train()
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    load_policy_into_vllm_instance(get_base_model(model), llm)
    final_accuracy = evaluate(llm, val_prompts, val_ground_truths)
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    
    if use_wandb:
        wandb.log({
            "eval/final_accuracy": final_accuracy,
            "eval_step": global_step,
        })
    
    return get_base_model(model), final_accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--val_data_path", type=str, default=VAL_DATA_PATH)
    parser.add_argument("--num_samples", type=int, default=7500,
                        help="Number of training samples (128, 256, 512, 1024, or 7500 for full)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_every_steps", type=int, default=25)
    parser.add_argument("--train_devices", type=str, default="0,1,2,3",
                        help="Comma-separated list of GPU IDs for training")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for vLLM inference (tensor parallelism)")
    parser.add_argument("--wandb_project", type=str, default="sft-math")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Wandb API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    # Parse train_devices
    train_devices = [int(x) for x in args.train_devices.split(",")]
    
    use_wandb = not args.no_wandb
    
    if use_wandb:
        # Login with API key if provided
        import os
        api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"sft-{args.num_samples}samples-lr{args.learning_rate}-bs{args.batch_size}",
        )
        setup_wandb_metrics()
    
    model, accuracy = train(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        eval_every_steps=args.eval_every_steps,
        train_devices=train_devices,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        use_wandb=use_wandb,
    )
    
    if use_wandb:
        wandb.finish()
    
    print(f"\nTraining complete. Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
