from __future__ import annotations

import random
import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from unittest.mock import patch
from vllm import LLM
from typing import Any


def vllm_set_random_seed(seed: int):
    """Set random seed for reproducibility (replaces vllm.model_executor.set_random_seed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    
    # Get the EOS token ID
    eos_token_id = tokenizer.eos_token_id
    # Use EOS as pad token if pad_token_id is not set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    
    # Tokenize each prompt and output separately (without special tokens)
    prompt_token_ids = []
    output_token_ids = []
    prompt_lens = []
    output_lens = []
    
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # Tokenize without special tokens
        prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(output_str, add_special_tokens=False)["input_ids"]
        
        prompt_token_ids.append(prompt_tokens)
        output_token_ids.append(output_tokens)
        prompt_lens.append(len(prompt_tokens))
        output_lens.append(len(output_tokens))
    
    # Calculate max length: prompt + output (no EOS added based on the snapshot analysis)
    # The max sequence is just prompt + output
    max_len = max(p_len + o_len for p_len, o_len in zip(prompt_lens, output_lens))
    
    # Build the full sequences with padding
    full_sequences = []
    for i in range(batch_size):
        # Concatenate prompt + output
        full_seq = prompt_token_ids[i] + output_token_ids[i]
        # Pad to max_len
        padding_needed = max_len - len(full_seq)
        full_seq = full_seq + [pad_token_id] * padding_needed
        full_sequences.append(full_seq)
    
    # Convert to tensor
    full_tensor = torch.tensor(full_sequences, dtype=torch.long)
    
    # input_ids: full sequence without the last token
    input_ids = full_tensor[:, :-1]
    
    # labels: full sequence without the first token (shifted by 1)
    labels = full_tensor[:, 1:]
    
    # response_mask: True for output tokens in labels, False for prompt/padding
    # In labels (shifted by 1), output tokens for example i are at positions:
    # [prompt_lens[i] - 1, prompt_lens[i] - 1 + output_lens[i] - 1] inclusive
    # i.e., [prompt_lens[i] - 1, prompt_lens[i] + output_lens[i] - 2]
    seq_len = input_ids.shape[1]
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    
    for i in range(batch_size):
        start_idx = prompt_lens[i] - 1
        end_idx = prompt_lens[i] + output_lens[i] - 2  # inclusive
        response_mask[i, start_idx:end_idx + 1] = True
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length). The entropy for each
            next-token prediction.

    Note: Use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    """
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    return (-normalized_logits * normalized_logits.exp()).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal language model,
    and optionally the entropy of the model's next-token distribution.

    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the
            correct device and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated
            prompt + response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced
            by your tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
                log p_theta(x_t | x_{<t}).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
                for each position (present only if return_token_entropy=True).
    """
    logits = model(input_ids).logits
    log_probs = logits.log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    entropy = None
    if return_token_entropy:
        entropy = compute_entropy(logits)
    return {
        "log_probs": log_probs,
        "token_entropy": entropy,
    }


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities
            from the SFT policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: The constant by which to divide the sum. It is fine to
            leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation.
                We return this so we can log it.
            metadata: Dict with metadata from the underlying loss call, and any other
                statistics you might want to log.

    Implementation tips:
        - You should call loss.backward() in this function. Make sure to adjust for
          gradient accumulation.
    """
    batch_size = policy_log_probs.shape[0]
    # SFT loss: negative log likelihood, mean over batch, scaled for gradient accumulation
    loss = -(policy_log_probs * response_mask).sum() / batch_size / normalize_constant / gradient_accumulation_steps
    loss.backward()
    return loss, {}


def log_generations(
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    rewards: list[dict[str, float]],
    token_entropies: list[float] | None = None,
    step: int | None = None,
    use_wandb: bool = True,
) -> dict:
    """Log generations from the model for monitoring during training.

    Args:
        prompts: List of input prompts.
        responses: List of responses generated by the SFT/RL model.
        ground_truths: List of ground-truth answers.
        rewards: List of reward dicts, each containing "reward", "format_reward", "answer_reward".
        token_entropies: Optional list of average token entropies per response.
        step: Optional training step for logging.
        use_wandb: Whether to log to wandb.

    Returns:
        dict with aggregated statistics:
            - avg_reward, avg_format_reward, avg_answer_reward
            - avg_response_length, avg_correct_response_length, avg_incorrect_response_length
            - avg_token_entropy (if token_entropies provided)
    """
    import wandb

    n = len(prompts)
    
    # Compute response lengths
    response_lengths = [len(r) for r in responses]
    
    # Separate correct vs incorrect based on answer_reward > 0
    correct_lengths = []
    incorrect_lengths = []
    for i, reward_dict in enumerate(rewards):
        if reward_dict.get("answer_reward", 0) > 0:
            correct_lengths.append(response_lengths[i])
        else:
            incorrect_lengths.append(response_lengths[i])
    
    # Compute statistics
    stats = {
        "avg_reward": sum(r["reward"] for r in rewards) / n if n > 0 else 0,
        "avg_format_reward": sum(r.get("format_reward", 0) for r in rewards) / n if n > 0 else 0,
        "avg_answer_reward": sum(r.get("answer_reward", 0) for r in rewards) / n if n > 0 else 0,
        "avg_response_length": sum(response_lengths) / n if n > 0 else 0,
        "avg_correct_response_length": sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0,
        "avg_incorrect_response_length": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0,
        "num_correct": len(correct_lengths),
        "num_incorrect": len(incorrect_lengths),
    }
    
    if token_entropies is not None:
        stats["avg_token_entropy"] = sum(token_entropies) / len(token_entropies) if token_entropies else 0
    
    # Create a table of examples for wandb
    if use_wandb and wandb.run is not None:
        # Log aggregated stats
        log_dict = {f"generations/{k}": v for k, v in stats.items()}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
        
        # Log a table of examples
        columns = ["prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        if token_entropies is not None:
            columns.append("token_entropy")
        
        data = []
        for i in range(n):
            row = [
                prompts[i],
                responses[i],
                ground_truths[i],
                rewards[i]["reward"],
                rewards[i].get("format_reward", 0),
                rewards[i].get("answer_reward", 0),
            ]
            if token_entropies is not None:
                row.append(token_entropies[i])
            data.append(row)
        
        table = wandb.Table(columns=columns, data=data)
        if step is not None:
            wandb.log({"generations/examples": table}, step=step)
        else:
            wandb.log({"generations/examples": table})
    
    return stats

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    
    In vLLM 0.14.x, device placement is controlled via CUDA_VISIBLE_DEVICES.
    This function expects CUDA_VISIBLE_DEVICES to be set appropriately before
    calling, such that cuda:0 in the visible devices is where vLLM should run.
    
    Args:
        model_id: Path to the model
        device: CUDA device string (e.g., "cuda:0"). Used for extracting GPU index.
        seed: Random seed
        gpu_memory_utilization: Fraction of GPU memory to use
    """
    import os
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    #     22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can place the vLLM model on the desired device
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    
    # Try to find and patch the profiling assertion if it exists
    # (path varies between vLLM versions)
    profiling_patches_to_try = [
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        "vllm.v1.worker.gpu_worker.GPUWorker._assert_memory_footprint_increased_during_profiling",
    ]
    
    profiling_patch = None
    for patch_path in profiling_patches_to_try:
        try:
            profiling_patch = patch(patch_path, return_value=None)
            # Test if the patch target exists
            profiling_patch.start()
            profiling_patch.stop()
            break
        except (AttributeError, ModuleNotFoundError):
            profiling_patch = None
            continue
    
    # Apply patches and create LLM
    if profiling_patch is not None:
        with world_size_patch, profiling_patch:
            return LLM(
                model=model_id,
                dtype=torch.bfloat16,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,  # Disable CUDA graphs to reduce memory
            )
    else:
        with world_size_patch:
            return LLM(
                model=model_id,
                dtype=torch.bfloat16,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,  # Disable CUDA graphs to reduce memory
            )

def init_vllm_multi_gpu(
    model_id: str,
    tensor_parallel_size: int = 1,
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """
    Initialize vLLM with multi-GPU support via tensor parallelism.
    
    In vLLM 0.14.x, device placement is controlled via CUDA_VISIBLE_DEVICES.
    Set CUDA_VISIBLE_DEVICES before calling this function to control which
    physical GPUs are used for tensor parallelism.
    
    Args:
        model_id: Path to the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        seed: Random seed
        gpu_memory_utilization: Fraction of GPU memory to use
    """
    vllm_set_random_seed(seed)
    
    # Monkeypatch from TRL to allow single-process vLLM
    # https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    
    # Try to find and patch the profiling assertion if it exists
    profiling_patches_to_try = [
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        "vllm.v1.worker.gpu_worker.GPUWorker._assert_memory_footprint_increased_during_profiling",
    ]
    
    profiling_patch = None
    for patch_path in profiling_patches_to_try:
        try:
            profiling_patch = patch(patch_path, return_value=None)
            profiling_patch.start()
            profiling_patch.stop()
            break
        except (AttributeError, ModuleNotFoundError):
            profiling_patch = None
            continue
    
    kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": torch.bfloat16,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,  # Disable CUDA graphs to reduce memory
    }
    
    if profiling_patch is not None:
        with world_size_patch, profiling_patch:
            return LLM(**kwargs)
    else:
        with world_size_patch:
            return LLM(**kwargs)

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    Load trained policy weights into vLLM instance.
    
    For vLLM 0.14.x V1 engine, we use apply_model to send a function that
    loads weights into the model running in the EngineCore subprocess.
    
    Adapted from TRL for vLLM 0.14.x compatibility.
    """
    state_dict = policy.state_dict()
    
    def _load_weights(model):
        """Function to load weights, called in the EngineCore subprocess."""
        model.load_weights(state_dict.items())
        return True
    
    # For V1 engine, use apply_model to load weights into subprocess
    if hasattr(llm.llm_engine, 'apply_model'):
        llm.llm_engine.apply_model(_load_weights)
    else:
        # Fallback for older vLLM versions (V0 engine)
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Parse the model output into a predicted option letter (i.e., 'A', 'B', 'C', or 'D').
    If the model output cannot be parsed into a prediction option letter, return None.
    """
    import re
    
    # Try to find answer patterns like "answer is A", "answer: B", "is C", etc.
    # Look for a standalone letter A, B, C, or D that appears to be the answer
    patterns = [
        r'(?:answer|choice)\s*(?:is|:)\s*([A-D])\b',  # "answer is A", "choice: B"
        r'\b([A-D])\s*(?:is\s+(?:the\s+)?(?:correct|right))',  # "A is correct", "B is the right"
        r'^([A-D])[\.\s]',  # Starts with "A." or "A "
        r'\b([A-D])\b',  # Any standalone letter A-D as fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """
    Parse GSM8K model output into a predicted numeric answer by taking the last number
    that occurs in the output.
    
    Args:
        model_output: The model's output text
        
    Returns:
        str with the predicted numeric answer, or None if no number found
    """
    import re
    
    # Find all numbers in the output (integers or decimals, possibly negative)
    # Match patterns like: 72, 3.14, -5, 1,000 (with commas)
    numbers = re.findall(r'-?[\d,]+\.?\d*', model_output)
    
    if not numbers:
        return None
    
    # Take the last number and clean it up (remove commas)
    last_number = numbers[-1].replace(',', '')
    
    # Validate it's actually a number
    try:
        float(last_number)
        return last_number
    except ValueError:
        return None


# ============================================================================
# PackedSFTDataset for instruction tuning
# ============================================================================

import os
import json
from pathlib import Path
from torch.utils.data import Dataset


class PackedSFTDataset(Dataset):
    """A PyTorch Dataset for instruction tuning that packs multiple documents
    into fixed-length sequences.
    
    This dataset:
    1. Loads instruction tuning data from a JSONL file
    2. Formats each example using an instruction prompt template
    3. Tokenizes all documents and concatenates them with BOS/EOS tokens
    4. Chunks the concatenated tokens into fixed-length sequences
    
    Each example contains:
    - input_ids: tokens [0:seq_length]
    - labels: tokens [1:seq_length+1] (shifted by 1 for next-token prediction)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        """Construct the dataset.
        
        Args:
            tokenizer: A transformers tokenizer for tokenizing and encoding.
            dataset_path: Path to instruction tuning data (JSONL format).
            seq_length: Desired length of sequences to generate.
            shuffle: Whether to shuffle documents before concatenation.
        """
        # Load the prompt template
        PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "alpaca_sft.prompt").read_text()
        
        # Load and format documents
        documents = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt = data["prompt"]
                response = data["response"]
                document = PROMPT_TEMPLATE.format(instruction=prompt, response=response)
                # Remove trailing newline to avoid tokenization issues
                document = document.rstrip('\n')
                documents.append(document)
        
        # Shuffle documents if requested
        if shuffle:
            random.shuffle(documents)
        
        # Tokenize all documents and concatenate with BOS/EOS tokens
        all_tokens = []
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        
        for doc in documents:
            # Add BOS token before each document
            all_tokens.append(bos_token_id)
            # Tokenize document without special tokens
            doc_tokens = tokenizer.encode(doc, add_special_tokens=False)
            all_tokens.extend(doc_tokens)
            # Add EOS token after each document
            all_tokens.append(eos_token_id)
        
        # Chunk into sequences of seq_length
        self.examples = []
        num_sequences = (len(all_tokens) - 1) // seq_length
        
        for i in range(num_sequences):
            start_idx = i * seq_length
            input_ids = all_tokens[start_idx : start_idx + seq_length]
            labels = all_tokens[start_idx + 1 : start_idx + seq_length + 1]
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
    
    def __len__(self) -> int:
        """Returns the number of sequences in this Dataset."""
        return len(self.examples)
    
    def __getitem__(self, i: int) -> dict[str, Tensor]:
        """Returns the ith element of the Dataset.
        
        Returns:
            dict with keys:
                - input_ids: tensor of shape (seq_length,)
                - labels: tensor of shape (seq_length,)
        """
        return self.examples[i]


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: A Dataset returning dicts with 'input_ids' and 'labels' tensors.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        
    Returns:
        A DataLoader that yields batches of stacked input_ids and labels.
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )