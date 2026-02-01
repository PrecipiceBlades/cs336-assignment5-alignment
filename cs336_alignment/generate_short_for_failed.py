"""
Generate short responses for samples that failed to generate short AND accurate responses.

This script:
1. Identifies questions that failed to regenerate (not in sft_short.jsonl)
2. Generates short responses for them (without accuracy requirement)
3. Combines with existing sft_short.jsonl to create final sft.jsonl
"""
import json
import os
import re
from pathlib import Path
from tqdm import tqdm

# Paths
CHECKPOINT_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/regenerate_checkpoint.json"
TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/train.jsonl"
SFT_SHORT_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft_short.jsonl"
OUTPUT_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/DeepSeek-R1-Distill-Qwen-32B"

# Load prompt template
PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "r1_zero.prompt").read_text()


def normalize_response_format(response: str) -> str:
    """Normalize the response format."""
    if not response.endswith("</answer>"):
        response = response + "</answer>"
    response = re.sub(r'</think>\s*.*?\s*<answer>', '</think> <answer>', response, flags=re.DOTALL)
    return response


def get_failed_questions():
    """Get questions that failed to regenerate short and accurate responses."""
    # Load questions already in sft_short.jsonl
    questions_in_sft_short = set()
    with open(SFT_SHORT_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions_in_sft_short.add(data["question"])
    
    # Load all training questions
    all_questions = {}
    with open(TRAIN_DATA_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            all_questions[data["problem"]] = data
    
    # Find failed questions (not in sft_short.jsonl)
    failed_questions = []
    for question, data in all_questions.items():
        if question not in questions_in_sft_short:
            failed_questions.append({
                "question": question,
                "solution": data["solution"],
            })
    
    return failed_questions


def generate_short_responses(
    failed_questions: list,
    max_response_tokens: int = 2048,
    tensor_parallel_size: int = 8,
    samples_per_question: int = 1,
):
    """Generate short responses for failed questions."""
    from vllm import LLM, SamplingParams
    
    if not failed_questions:
        print("No failed questions to process!")
        return []
    
    print(f"Generating short responses for {len(failed_questions)} questions...")
    
    # Initialize model
    print(f"Loading model from {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.60,
        max_num_seqs=64,
        enable_prefix_caching=True,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_response_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=samples_per_question,
    )
    
    # Generate
    prompts = [PROMPT_TEMPLATE.format(question=q["question"]) for q in failed_questions]
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect results
    results = []
    for question_data, output in tqdm(zip(failed_questions, outputs), desc="Processing", total=len(failed_questions)):
        question = question_data["question"]
        
        if output.outputs:
            response = output.outputs[0].text
            normalized_response = normalize_response_format(response)
            results.append({
                "question": question,
                "response": normalized_response,
            })
        else:
            print(f"Warning: No output for question: {question[:50]}...")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_response_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--samples_per_question", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true", help="Just show what would be done")
    args = parser.parse_args()
    
    # Get failed questions
    print("Identifying failed questions...")
    failed_questions = get_failed_questions()
    print(f"Found {len(failed_questions)} failed questions")
    
    if args.dry_run:
        print("\n[DRY RUN] Would generate short responses for these questions:")
        for i, q in enumerate(failed_questions[:5]):
            print(f"  {i+1}. {q['question'][:80]}...")
        if len(failed_questions) > 5:
            print(f"  ... and {len(failed_questions) - 5} more")
        return
    
    # Generate short responses
    new_samples = generate_short_responses(
        failed_questions,
        args.max_response_tokens,
        args.tensor_parallel_size,
        args.samples_per_question,
    )
    
    print(f"\nGenerated {len(new_samples)} new samples")
    
    # Load existing sft_short.jsonl
    print(f"\nLoading existing data from {SFT_SHORT_PATH}...")
    existing_samples = []
    with open(SFT_SHORT_PATH, 'r') as f:
        for line in f:
            existing_samples.append(json.loads(line))
    print(f"Loaded {len(existing_samples)} existing samples")
    
    # Combine
    all_samples = existing_samples + new_samples
    print(f"Total samples: {len(all_samples)}")
    
    # Save final output
    print(f"\nSaving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("Final Dataset Statistics:")
    print(f"  Existing samples from sft_short.jsonl: {len(existing_samples)}")
    print(f"  New samples (short, no accuracy req): {len(new_samples)}")
    print(f"  Total samples in sft.jsonl: {len(all_samples)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
