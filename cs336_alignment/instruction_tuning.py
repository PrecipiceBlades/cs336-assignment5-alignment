"""
Instruction Tuning Script for Llama 3.1 8B Base Model with FSDP Support.

Fine-tunes the Llama 3.1 8B base model on instruction tuning data using:
- Context length: 512 tokens
- Total batch size: 32 sequences per gradient step
- Micro batch size: 2 (with FlashAttention-2 and bfloat16)
- Learning rate: 2e-5 with cosine decay
- Linear warmup: 3% of total training steps
- Single epoch training
- Multi-GPU training with FSDP (Fully Sharded Data Parallel)

Usage (single GPU):
    python -m cs336_alignment.instruction_tuning \
        --model_path /path/to/llama-3.1-8b \
        --train_data_path /path/to/train.jsonl \
        --output_dir /path/to/output

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 -m cs336_alignment.instruction_tuning \
        --model_path /path/to/llama-3.1-8b \
        --train_data_path /path/to/train.jsonl \
        --output_dir /path/to/output
"""
import os
import sys
import math
import random
import functools
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm


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


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


def collate_fn(batch):
    """Collate function for DataLoader to stack tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def get_fsdp_config():
    """Get FSDP configuration with mixed precision."""
    # Mixed precision policy for bfloat16
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto wrap policy for transformer layers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    return {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
        "use_orig_params": True,  # Required for gradient checkpointing
    }


def compute_validation_loss(model, val_dataloader, device, is_distributed=False):
    """Compute average validation loss (all ranks must participate for FSDP)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += labels.numel()
    
    # Aggregate across all ranks if distributed
    if is_distributed:
        loss_tensor = torch.tensor([total_loss], device=device)
        tokens_tensor = torch.tensor([total_tokens], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor.item()
        total_tokens = int(tokens_tensor.item())
    
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def save_fsdp_model(model, tokenizer, output_dir, rank):
    """Save FSDP model by gathering full state dict on rank 0."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    
    # Configure to gather full state dict on rank 0
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()
        
        if rank == 0:
            # Get the underlying model config
            # We need to save a HuggingFace compatible checkpoint
            print_flush(f"Saving model to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Load a fresh model to get the config, then load state dict
            from transformers import AutoConfig
            config = model._fsdp_wrapped_module.config
            
            # Save config
            config.save_pretrained(output_dir)
            
            # Save state dict
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            print_flush("Model saved successfully!")


def train(
    model_path: str,
    train_data_path: str,
    output_dir: str,
    val_data_path: str = None,
    seq_length: int = 512,
    micro_batch_size: int = 2,
    total_batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.03,
    num_epochs: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    eval_interval: int = 100,
    save_interval: int = None,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_project: str = "instruction-tuning",
):
    """Run instruction tuning training with FSDP support.
    
    Args:
        model_path: Path to pretrained model
        train_data_path: Path to training data (JSONL format)
        output_dir: Directory to save the fine-tuned model
        val_data_path: Optional path to validation data
        seq_length: Context length for training
        micro_batch_size: Batch size per forward pass per GPU
        total_batch_size: Total batch size per gradient step (across all GPUs)
        learning_rate: Peak learning rate
        warmup_ratio: Fraction of steps for linear warmup
        num_epochs: Number of training epochs
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Steps between logging
        eval_interval: Steps between validation
        save_interval: Steps between checkpoints (None = only save at end)
        seed: Random seed
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
    """
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    set_random_seed(seed + rank)  # Different seed per process for data shuffling
    
    # Calculate gradient accumulation steps (accounting for multiple GPUs)
    per_gpu_batch_size = micro_batch_size
    assert total_batch_size % (per_gpu_batch_size * world_size) == 0, \
        f"total_batch_size ({total_batch_size}) must be divisible by per_gpu_batch_size * world_size ({per_gpu_batch_size * world_size})"
    gradient_accumulation_steps = total_batch_size // (per_gpu_batch_size * world_size)
    
    if is_main_process(rank):
        print_flush(f"\n{'='*60}")
        print_flush("Instruction Tuning Configuration (FSDP)")
        print_flush(f"{'='*60}")
        print_flush(f"Model: {model_path}")
        print_flush(f"Train data: {train_data_path}")
        print_flush(f"Output dir: {output_dir}")
        print_flush(f"Sequence length: {seq_length}")
        print_flush(f"Micro batch size per GPU: {micro_batch_size}")
        print_flush(f"Total batch size: {total_batch_size}")
        print_flush(f"World size (num GPUs): {world_size}")
        print_flush(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print_flush(f"Learning rate: {learning_rate}")
        print_flush(f"Warmup ratio: {warmup_ratio}")
        print_flush(f"Epochs: {num_epochs}")
        print_flush(f"Device: {device}")
        print_flush(f"Distributed: {is_distributed}")
        print_flush(f"{'='*60}\n")
    
    # Initialize wandb only on main process
    if use_wandb and is_main_process(rank):
        import wandb
        wandb.init(
            project=wandb_project,
            config={
                "model_path": model_path,
                "train_data_path": train_data_path,
                "seq_length": seq_length,
                "micro_batch_size": micro_batch_size,
                "total_batch_size": total_batch_size,
                "world_size": world_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "num_epochs": num_epochs,
                "max_grad_norm": max_grad_norm,
                "seed": seed,
                "parallel_mode": "FSDP",
            },
            name=f"sft-llama-lr{learning_rate}-bs{total_batch_size}-{world_size}gpu-fsdp",
        )
    
    # Load tokenizer first (needed for data loading)
    if is_main_process(rank):
        print_flush("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load training data using PackedSFTDataset (before model to save memory)
    if is_main_process(rank):
        print_flush("\nLoading training data...")
    
    from .utils import PackedSFTDataset
    
    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=train_data_path,
        seq_length=seq_length,
        shuffle=True,
    )
    
    if is_main_process(rank):
        print_flush(f"Training dataset: {len(train_dataset)} sequences")
    
    # Use DistributedSampler for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if is_distributed else None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Load validation data if provided (all ranks need this for FSDP)
    val_dataloader = None
    if val_data_path:
        if is_main_process(rank):
            print_flush("Loading validation data...")
        val_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=val_data_path,
            seq_length=seq_length,
            shuffle=False,
        )
        if is_main_process(rank):
            print_flush(f"Validation dataset: {len(val_dataset)} sequences")
        
        # Use DistributedSampler for validation too (FSDP requires all ranks)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        ) if is_distributed else None
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
    
    # Load model
    if is_main_process(rank):
        print_flush("\nLoading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,  # Disable KV cache for training
    )
    
    # Enable gradient checkpointing before FSDP wrapping
    model.gradient_checkpointing_enable()
    if is_main_process(rank):
        print_flush("Gradient checkpointing enabled")
    
    # Wrap model with FSDP
    if is_distributed:
        if is_main_process(rank):
            print_flush("Wrapping model with FSDP...")
        
        fsdp_config = get_fsdp_config()
        model = FSDP(model, **fsdp_config)
        
        if is_main_process(rank):
            print_flush("Model wrapped with FSDP")
    else:
        model = model.to(device)
    
    if is_main_process(rank):
        print_flush(f"Model loaded on {device}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(local_rank) / 1e9
            reserved = torch.cuda.memory_reserved(local_rank) / 1e9
            print_flush(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    if is_main_process(rank):
        print_flush(f"\nTraining plan:")
        print_flush(f"  Batches per epoch per GPU: {len(train_dataloader)}")
        print_flush(f"  Steps per epoch: {steps_per_epoch}")
        print_flush(f"  Total steps: {total_steps}")
        print_flush(f"  Warmup steps: {warmup_steps}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Initial validation to verify FSDP works correctly
    if val_dataloader:
        if is_main_process(rank):
            print_flush("\nRunning initial validation to verify FSDP setup...")
        
        initial_val_loss = compute_validation_loss(model, val_dataloader, device, is_distributed)
        
        if is_main_process(rank):
            print_flush(f"Initial validation loss: {initial_val_loss:.4f}")
            if use_wandb:
                import wandb
                wandb.log({
                    "val/loss": initial_val_loss,
                    "val/step": 0,
                })
        
        # Sync all ranks after initial validation
        if is_distributed:
            dist.barrier()
        
        if is_main_process(rank):
            print_flush("Initial validation completed successfully!")
    
    # Training loop
    if is_main_process(rank):
        print_flush("\nStarting training...")
    
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    accumulated_steps = 0
    
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        if is_main_process(rank):
            print_flush(f"\n{'='*60}")
            print_flush(f"Epoch {epoch + 1}/{num_epochs}")
            print_flush(f"{'='*60}")
        
        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=not is_main_process(rank))
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='mean'
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            accumulated_steps += 1
            
            # Gradient step
            if accumulated_steps % gradient_accumulation_steps == 0:
                # Gradient clipping (FSDP handles this differently)
                if max_grad_norm is not None and max_grad_norm > 0:
                    if is_distributed:
                        model.clip_grad_norm_(max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Compute average loss for this gradient step
                avg_loss = accumulated_loss / gradient_accumulation_steps
                accumulated_loss = 0.0
                
                # Update progress bar (only on main process)
                if is_main_process(rank):
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    })
                
                # Logging (only on main process)
                if is_main_process(rank) and global_step % log_interval == 0:
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + (batch_idx + 1) / len(train_dataloader),
                            "train/step": global_step,
                        })
                
                # Validation - all ranks must participate for FSDP
                if val_dataloader and global_step % eval_interval == 0:
                    val_loss = compute_validation_loss(model, val_dataloader, device, is_distributed)
                    
                    if is_main_process(rank):
                        print_flush(f"\nStep {global_step}: val_loss = {val_loss:.4f}")
                        
                        if use_wandb:
                            import wandb
                            wandb.log({
                                "val/loss": val_loss,
                                "val/step": global_step,
                            })
                    
                    model.train()
                
                # Checkpoint saving
                if save_interval and global_step % save_interval == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    if is_main_process(rank):
                        print_flush(f"\nSaving checkpoint to {checkpoint_dir}")
                    save_fsdp_model(model, tokenizer, checkpoint_dir, rank)
                    
                    # Sync all processes after checkpoint
                    if is_distributed:
                        dist.barrier()
    
    # Final validation - all ranks must participate for FSDP
    if val_dataloader:
        final_val_loss = compute_validation_loss(model, val_dataloader, device, is_distributed)
        
        if is_main_process(rank):
            print_flush(f"\nFinal validation loss: {final_val_loss:.4f}")
            
            if use_wandb:
                import wandb
                wandb.log({
                    "val/final_loss": final_val_loss,
                })
    
    # Save final model
    if is_main_process(rank):
        print_flush(f"\nSaving final model to {output_dir}")
    
    save_fsdp_model(model, tokenizer, output_dir, rank)
    
    # Sync before cleanup
    if is_distributed:
        dist.barrier()
    
    if is_main_process(rank):
        if use_wandb:
            import wandb
            wandb.finish()
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process(rank):
        print_flush(f"\n{'='*60}")
        print_flush("Training complete!")
        print_flush(f"{'='*60}")
    
    return model, tokenizer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Instruction tuning for Llama 3.1 8B with FSDP support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained Llama 3.1 8B base model")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data (JSONL format with 'prompt' and 'response' fields)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    
    # Optional arguments
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data (optional)")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Context length for training (default: 512)")
    parser.add_argument("--micro_batch_size", type=int, default=2,
                        help="Batch size per forward pass per GPU (default: 2)")
    parser.add_argument("--total_batch_size", type=int, default=32,
                        help="Total batch size per gradient step (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Peak learning rate (default: 2e-5)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Fraction of steps for linear warmup (default: 0.03)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (default: 1.0)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Steps between logging (default: 10)")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Steps between validation (default: 100)")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Steps between checkpoints (default: only save at end)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="instruction-tuning",
                        help="Wandb project name (default: instruction-tuning)")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Wandb API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    
    args = parser.parse_args()
    
    use_wandb = not args.no_wandb
    
    # Only login to wandb on main process
    rank = int(os.environ.get("RANK", 0))
    if use_wandb and rank == 0:
        import wandb
        api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
    
    train(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        val_data_path=args.val_data_path,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        total_batch_size=args.total_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
