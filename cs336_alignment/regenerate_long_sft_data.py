"""
Regenerate SFT data for samples with long responses using rejection sampling.

The original SFT data was generated with max_tokens=16384, but evaluation uses max_tokens=2048.
This script:
1. Identifies samples with response tokens > 2048
2. Regenerates these samples with a shorter max_tokens limit
3. Uses rejection sampling to get correct (format + answer) responses
4. Saves the new dataset with shorter responses
"""
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
ORIGINAL_SFT_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft_short.jsonl"

# Load prompt template
PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "r1_zero.prompt").read_text()


def normalize_response_format(response: str) -> str:
    """
    Normalize the response format so r1_zero_reward_fn can parse it.
    DeepSeek R1 outputs: </think>\n\n...summary...\n\n<answer>
    But r1_zero_reward_fn expects: </think> <answer>
    """
    if not response.endswith("</answer>"):
        response = response + "</answer>"
    response = re.sub(r'</think>\s*.*?\s*<answer>', '</think> <answer>', response, flags=re.DOTALL)
    return response


def load_original_data_and_identify_long_samples(
    sft_path: str,
    tokenizer,
    max_response_tokens: int = 2048,
):
    """Load original SFT data and identify samples that need regeneration."""
    short_samples = []  # Keep as-is
    long_samples = []   # Need regeneration
    
    with open(sft_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Analyzing samples")):
            data = json.loads(line)
            question = data["question"]
            response = data["response"]
            
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            item = {
                "idx": i,
                "question": question,
                "original_response": response,
                "response_tokens": len(response_tokens),
            }
            
            if len(response_tokens) <= max_response_tokens:
                # Keep short samples
                short_samples.append({
                    "question": question,
                    "response": response,
                })
            else:
                # Need to regenerate
                long_samples.append(item)
    
    return short_samples, long_samples


def extract_ground_truth(question: str, train_data_path: str):
    """Extract ground truth for a question from training data."""
    from drgrpo_grader import extract_boxed_answer
    
    with open(train_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data["problem"] == question:
                return extract_boxed_answer(data["solution"])
    return None


def regenerate_long_samples(
    long_samples: list,
    train_data_path: str,
    model_path: str,
    max_response_tokens: int = 2048,
    max_attempts: int = 8,
    tensor_parallel_size: int = 8,
    samples_per_attempt: int = 4,  # Generate multiple samples per attempt
    checkpoint_path: str = None,  # Path to save checkpoints
):
    """Regenerate responses for long samples using rejection sampling with shorter max_tokens."""
    from vllm import LLM, SamplingParams
    from .drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn
    
    # Load ground truths
    print("Loading ground truths from training data...")
    gt_map = {}
    with open(train_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gt = extract_boxed_answer(data["solution"])
            gt_map[data["problem"]] = gt
    
    # Add ground truths to long samples
    for sample in long_samples:
        sample["ground_truth"] = gt_map.get(sample["question"])
    
    # Filter samples with ground truth
    samples_with_gt = [s for s in long_samples if s["ground_truth"] is not None]
    print(f"Samples with ground truth: {len(samples_with_gt)}/{len(long_samples)}")
    
    # Check for existing checkpoint
    results = []
    completed_questions = set()
    start_attempt = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            completed_questions = set(checkpoint_data.get("completed_questions", []))
            start_attempt = checkpoint_data.get("last_attempt", 0) + 1
        print(f"Resumed with {len(results)} already regenerated samples, starting from attempt {start_attempt}")
    
    # Filter out already completed samples
    original_count = len(samples_with_gt)
    samples_with_gt = [s for s in samples_with_gt if s["question"] not in completed_questions]
    print(f"Remaining samples to process: {len(samples_with_gt)} (skipped {original_count - len(samples_with_gt)})")
    
    if len(samples_with_gt) == 0:
        print("All samples already processed from checkpoint!")
        # Return failed as empty since we don't track failed in checkpoint
        return results, []
    
    # Initialize model
    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.60,  # Reduced to work with partial GPU memory
        max_num_seqs=64,  # Reduced for memory
        enable_prefix_caching=True,
    )
    
    # Sampling parameters with shorter max_tokens
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_response_tokens,  # Shorter limit!
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=samples_per_attempt,  # Generate multiple samples
    )
    
    # Track results
    pending = samples_with_gt.copy()
    
    def save_checkpoint(attempt_num):
        """Save current progress to checkpoint file."""
        if checkpoint_path:
            checkpoint_data = {
                "results": results,
                "completed_questions": list(completed_questions),
                "pending_count": len(pending),
                "last_attempt": attempt_num,
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"[Checkpoint] Saved: {len(results)} results, {len(pending)} pending, attempt {attempt_num}")
    
    for attempt in range(start_attempt, max_attempts + 1):
        if not pending:
            break
        
        print(f"\n{'='*60}")
        print(f"Rejection Sampling - Attempt {attempt}/{max_attempts}")
        print(f"Pending questions: {len(pending)}")
        print(f"{'='*60}")
        
        # Generate
        prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in pending]
        print(f"Generating {len(prompts)} x {samples_per_attempt} responses...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Grade
        new_pending = []
        new_successes = 0
        
        for sample, output in tqdm(zip(pending, outputs), desc="Grading", total=len(pending)):
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            
            # Try each generated sample
            success = False
            best_response = None
            best_response_len = float('inf')
            
            for gen_output in output.outputs:
                response = gen_output.text
                normalized_response = normalize_response_format(response)
                
                reward_info = r1_zero_reward_fn(normalized_response, ground_truth, fast=False)
                
                if reward_info["format_reward"] == 1.0 and reward_info["answer_reward"] == 1.0:
                    # Success! Keep the shortest correct response
                    if len(normalized_response) < best_response_len:
                        best_response = normalized_response
                        best_response_len = len(normalized_response)
                    success = True
            
            if success and best_response:
                results.append({
                    "question": question,
                    "response": best_response,
                })
                completed_questions.add(question)
                new_successes += 1
            else:
                # Failed, try again
                new_pending.append(sample)
        
        success_rate = new_successes / len(pending) * 100 if pending else 0
        print(f"Attempt {attempt} results: {new_successes}/{len(pending)} ({success_rate:.1f}%)")
        print(f"Total regenerated so far: {len(results)}")
        
        pending = new_pending
        
        # Save checkpoint after each attempt
        save_checkpoint(attempt)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Regeneration Complete!")
    print(f"Successfully regenerated: {len(results)}/{original_count}")
    print(f"Failed (will use original or exclude): {len(pending)}")
    
    return results, pending


def main():
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_sft_path", type=str, default=ORIGINAL_SFT_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--train_data_path", type=str, 
                        default="/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--max_response_tokens", type=int, default=2048,
                        help="Maximum response tokens (shorter than original 16384)")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--max_attempts", type=int, default=8)
    parser.add_argument("--samples_per_attempt", type=int, default=4)
    parser.add_argument("--keep_failed_original", action="store_true",
                        help="Keep original long responses for failed regenerations (default: exclude)")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/regenerate_checkpoint.json",
                        help="Path to save/load checkpoint for resuming")
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    )
    
    # Identify short vs long samples
    print(f"\nAnalyzing original SFT data: {args.original_sft_path}")
    short_samples, long_samples = load_original_data_and_identify_long_samples(
        args.original_sft_path,
        tokenizer,
        args.max_response_tokens,
    )
    
    print(f"\nShort samples (keep as-is): {len(short_samples)}")
    print(f"Long samples (need regeneration): {len(long_samples)}")
    
    if len(long_samples) == 0:
        print("No long samples to regenerate. Copying original data.")
        with open(args.output_path, 'w') as f:
            for sample in short_samples:
                f.write(json.dumps(sample) + "\n")
        return
    
    # Regenerate long samples
    regenerated, failed = regenerate_long_samples(
        long_samples,
        args.train_data_path,
        args.model_path,
        args.max_response_tokens,
        args.max_attempts,
        args.tensor_parallel_size,
        args.samples_per_attempt,
        args.checkpoint_path,
    )
    
    # Combine results
    print(f"\nCombining results...")
    all_samples = short_samples + regenerated
    
    if args.keep_failed_original:
        print(f"Adding {len(failed)} failed samples with original responses...")
        for sample in failed:
            all_samples.append({
                "question": sample["question"],
                "response": sample["original_response"],
            })
    else:
        print(f"Excluding {len(failed)} failed samples (no short correct response found)")
    
    # Save
    print(f"\nSaving {len(all_samples)} samples to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("Final Dataset Statistics:")
    print(f"  Original short samples: {len(short_samples)}")
    print(f"  Successfully regenerated: {len(regenerated)}")
    if args.keep_failed_original:
        print(f"  Kept original (failed regeneration): {len(failed)}")
    else:
        print(f"  Excluded (failed regeneration): {len(failed)}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
