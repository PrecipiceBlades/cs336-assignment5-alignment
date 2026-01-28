"""
Generate SFT data using DeepSeek R1 on MATH training set with Rejection Sampling.

This script:
1. Loads MATH training data (7.5k problems)
2. Generates reasoning traces using DeepSeek R1
3. Uses rejection sampling: retry failed questions up to max_attempts times
4. Saves correct responses to sft.jsonl
"""
import json
import os
import re
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn

# Paths
TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/train.jsonl"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"

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


def generate_sft_data_rejection_sampling(
    train_data_path: str,
    model_path: str,
    output_path: str,
    tensor_parallel_size: int = 8,
    max_attempts: int = 8,  # Maximum rejection sampling attempts per question
):
    """Generate SFT data using rejection sampling."""
    
    # Load training data
    print("Loading training data...")
    all_data = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            data = json.loads(line)
            all_data.append({
                "question": data["problem"],
                "ground_truth": extract_boxed_answer(data["solution"]),
            })
    
    print(f"Loaded {len(all_data)} training examples")
    
    # Check for existing progress (resume support)
    processed_questions = {}  # question -> response
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                processed_questions[item["question"]] = item["response"]
        print(f"Found {len(processed_questions)} already processed examples, resuming...")
    
    # Filter out already processed
    remaining_data = [d for d in all_data if d["question"] not in processed_questions]
    print(f"Remaining to process: {len(remaining_data)} examples")
    
    if len(remaining_data) == 0:
        print("All examples already processed!")
        return
    
    # Initialize model
    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,  # Balanced for memory safety
        enable_prefix_caching=True,
    )
    
    # Sampling parameters - single sample per prompt
    sampling_params = SamplingParams(
        temperature=0.7,  # Slightly higher for diversity across attempts
        top_p=0.95,
        max_tokens=16384,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1,  # Single sample per attempt
    )
    
    # Track pending questions (those that haven't succeeded yet)
    pending = remaining_data.copy()
    results = []  # Successful results
    attempt_stats = {i: 0 for i in range(1, max_attempts + 1)}  # Count per attempt number
    
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
            
        print(f"\n{'='*60}")
        print(f"Rejection Sampling - Attempt {attempt}/{max_attempts}")
        print(f"Pending questions: {len(pending)}")
        print(f"{'='*60}")
        
        # Generate for all pending questions
        prompts = [PROMPT_TEMPLATE.format(question=d["question"]) for d in pending]
        
        print(f"Generating {len(prompts)} responses...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Grade responses
        new_pending = []
        new_successes = 0
        
        for data_item, output in tqdm(zip(pending, outputs), desc="Grading", total=len(pending)):
            question = data_item["question"]
            ground_truth = data_item["ground_truth"]
            
            response = output.outputs[0].text
            normalized_response = normalize_response_format(response)
            
            reward_info = r1_zero_reward_fn(normalized_response, ground_truth, fast=False)
            
            if reward_info["format_reward"] == 1.0 and reward_info["answer_reward"] == 1.0:
                # Success! Save this response
                results.append({
                    "question": question,
                    "response": normalized_response,
                })
                attempt_stats[attempt] += 1
                new_successes += 1
            else:
                # Failed, add to pending for next attempt
                new_pending.append(data_item)
        
        # Save successful results incrementally
        if new_successes > 0:
            with open(output_path, "a") as f:
                for item in results[-new_successes:]:
                    f.write(json.dumps(item) + "\n")
        
        success_rate = new_successes / len(pending) * 100 if pending else 0
        total_success = len(processed_questions) + len(results)
        total_questions = len(all_data)
        
        print(f"\nAttempt {attempt} results:")
        print(f"  New successes: {new_successes}/{len(pending)} ({success_rate:.1f}%)")
        print(f"  Running total: {total_success}/{total_questions} ({total_success/total_questions*100:.1f}%)")
        print(f"  Remaining: {len(new_pending)}")
        
        pending = new_pending
    
    # Final summary
    total_success = len(processed_questions) + len(results)
    total_questions = len(all_data)
    
    print(f"\n{'='*60}")
    print(f"Rejection Sampling Complete!")
    print(f"{'='*60}")
    print(f"Total questions: {total_questions}")
    print(f"Total successes: {total_success} ({total_success/total_questions*100:.2f}%)")
    print(f"Failed after {max_attempts} attempts: {len(pending)}")
    print(f"\nSuccess distribution by attempt:")
    for attempt, count in attempt_stats.items():
        if count > 0:
            print(f"  Attempt {attempt}: {count} ({count/len(remaining_data)*100:.1f}%)")
    print(f"\nSFT data saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                        help="Number of GPUs for tensor parallelism (default: 8)")
    parser.add_argument("--max_attempts", type=int, default=8,
                        help="Maximum rejection sampling attempts per question (default: 8)")
    args = parser.parse_args()
    
    generate_sft_data_rejection_sampling(
        train_data_path=args.train_data_path,
        model_path=args.model_path,
        output_path=args.output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
