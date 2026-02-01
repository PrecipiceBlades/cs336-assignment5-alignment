"""
SFT (Supervised Fine-Tuning) on MATH dataset.

Run SFT on reasoning examples using Qwen 2.5 Math 1.5B base model.

Device Isolation Strategy:
- vLLM GPU is controlled by VLLM_GPU environment variable (physical GPU index)
- Training GPU is controlled by TRAIN_GPU environment variable (physical GPU index)
- The script sets CUDA_VISIBLE_DEVICES dynamically to achieve isolation

Usage:
    python -m cs336_alignment.sft --vllm_gpu 0 --train_gpu 1
"""
import os
import sys
import json
import random
import numpy as np
from pathlib import Path


def print_flush(*args, **kwargs):
    """Print with immediate flush for subprocess compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Paths
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "r1_zero.prompt").read_text()


def load_train_data(data_path: str, num_samples: int = None):
    """Load training data."""
    prompts = []
    responses = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            # Remove <think> prefix from response since prompt ends with <think>
            response = data["response"].removeprefix("<think>")
            # Keep <think> in prompt to match validation format
            prompt = PROMPT_TEMPLATE.format(question=question)
            prompts.append(prompt)
            responses.append(response)
            if num_samples and len(prompts) >= num_samples:
                break
    return prompts, responses


def load_val_data(data_path: str):
    """Load validation data."""
    from .drgrpo_grader import extract_boxed_answer
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


def setup_wandb_metrics():
    """Setup wandb metrics for train and eval steps."""
    import wandb
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def evaluate(llm, sampling_params, val_prompts: list[str], val_ground_truths: list[str]) -> float:
    """Evaluate model using vLLM and return accuracy."""
    from .drgrpo_grader import r1_zero_reward_fn
    
    outputs = llm.generate(val_prompts, sampling_params)
    
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
    vllm_gpu: int = 0,
    train_gpu: int = 1,
    vllm_gpu_memory_utilization: float = 0.5,
    use_wandb: bool = True,
):
    """Run SFT training with vLLM for evaluation.
    
    Device Strategy:
    - If CUDA_VISIBLE_DEVICES is set externally, use the first visible GPU (cuda:0)
    - Otherwise, set CUDA_VISIBLE_DEVICES to vllm_gpu
    - vLLM and training model share the same GPU
    - vllm_gpu_memory_utilization controls vLLM's memory, leaving rest for training
    """
    # Check if CUDA_VISIBLE_DEVICES is already set externally
    external_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if external_cuda_visible:
        # Use externally set GPU (will be cuda:0)
        print_flush(f"Using externally set CUDA_VISIBLE_DEVICES={external_cuda_visible}")
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
    
    # Now import torch and vllm - they will only see the vLLM GPU
    import torch
    import wandb
    from tqdm import tqdm
    from vllm import SamplingParams
    
    from .utils import (
        init_vllm,
        load_policy_into_vllm_instance,
        tokenize_prompt_and_output,
        get_response_log_probs,
        sft_microbatch_train_step,
    )
    
    # Sampling parameters for vLLM generation
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=2048,
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
    
    # Load training model on cuda:0 (same GPU as vLLM)
    # vLLM's gpu_memory_utilization controls how much it uses, leaving rest for training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
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
    print_flush(f"\nLoading training data ({num_samples} samples)...")
    train_prompts, train_responses = load_train_data(train_data_path, num_samples)
    print_flush(f"Loaded {len(train_prompts)} training examples")
    
    print_flush("Loading validation data...")
    val_prompts, val_ground_truths = load_val_data(val_data_path)
    print_flush(f"Loaded {len(val_prompts)} validation examples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    global_step = 0
    model.train()
    
    for epoch in range(num_epochs):
        print_flush(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for i in tqdm(range(0, len(train_prompts), batch_size), desc=f"Epoch {epoch+1}"):
            batch_prompts = train_prompts[i:i+batch_size]
            batch_responses = train_responses[i:i+batch_size]
            
            # Tokenize
            items = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = items["input_ids"].to("cuda:0")
            labels = items["labels"].to("cuda:0")
            response_mask = items["response_mask"].to("cuda:0")
            
            # Forward pass
            log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
            
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
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print_flush(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    print_flush("\nFinal evaluation...")
    model.eval()
    load_policy_into_vllm_instance(model, llm)
    final_accuracy = evaluate(llm, sampling_params, val_prompts, val_ground_truths)
    print_flush(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    
    if use_wandb:
        wandb.log({
            "eval/final_accuracy": final_accuracy,
            "eval_step": global_step,
        })
    
    return model, final_accuracy


def main():
    import argparse
    import wandb
    
    parser = argparse.ArgumentParser(
        description="SFT training with vLLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Device Configuration:
  vLLM and training model share the same GPU to avoid device isolation issues.
  The vllm_gpu_memory_utilization controls how much memory vLLM uses,
  leaving the rest for training.
  
  Example:
    python -m cs336_alignment.sft --vllm_gpu 0 --vllm_gpu_memory_utilization 0.5
    
  With gpu_memory_utilization=0.5 on an 80GB GPU:
    - vLLM uses ~40GB for KV cache
    - Training model uses ~3GB (Qwen 1.5B in bf16)
    - Remaining ~37GB for gradients, activations, optimizer states
        """
    )
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--val_data_path", type=str, default=VAL_DATA_PATH)
    parser.add_argument("--num_samples", type=int, default=7500,
                        help="Number of training samples (128, 256, 512, 1024, or 7500 for full)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--vllm_gpu", type=int, default=0,
                        help="Physical GPU index for vLLM and training (default: 0)")
    parser.add_argument("--train_gpu", type=int, default=1,
                        help="(Deprecated) Training now shares GPU with vLLM")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5,
                        help="Fraction of GPU memory for vLLM KV cache (default: 0.5, balances inference speed and training memory)")
    parser.add_argument("--wandb_project", type=str, default="sft-math")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Wandb API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    use_wandb = not args.no_wandb
    
    if use_wandb:
        # Login with API key if provided
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
        vllm_gpu=args.vllm_gpu,
        train_gpu=args.train_gpu,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        use_wandb=use_wandb,
    )
    
    if use_wandb:
        wandb.finish()
    
    print_flush(f"\nTraining complete. Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
