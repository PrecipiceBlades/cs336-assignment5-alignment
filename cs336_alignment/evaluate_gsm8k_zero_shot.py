"""
Evaluate Llama 3.1 8B zero-shot performance on GSM8K.

This script:
1. Loads the GSM8K test data
2. Formats them as prompts to the language model
3. Generates outputs for each example using vLLM
4. Parses model outputs to extract predicted numeric answers
5. Calculates evaluation metrics (accuracy)
6. Serializes examples, model generations, and evaluation scores to disk

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python -m cs336_alignment.evaluate_gsm8k_zero_shot
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# Parse --gpu argument early, before any CUDA imports
def _setup_gpu():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"Using GPU {gpu}")
            return
        elif arg.startswith("--gpu="):
            gpu = arg.split("=")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"Using GPU {gpu}")
            return

_setup_gpu()

from tqdm import tqdm
from vllm import LLM, SamplingParams

from .utils import parse_gsm8k_response

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "gsm8k"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Llama-3.1-8B"

# GSM8K prompt template - simple format as specified in assignment
GSM8K_PROMPT_TEMPLATE = """{question}
Answer:"""


def load_gsm8k_data(data_dir: Path, split: str = "test") -> list[dict]:
    """
    Load GSM8K data from JSONL file.
    
    Args:
        data_dir: Path to GSM8K data directory
        split: Which split to load ("train" or "test")
    
    Returns:
        List of GSM8K examples with keys: question, answer, ground_truth (numeric)
    """
    data_file = data_dir / f"{split}.jsonl"
    examples = []
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Extract the numeric answer after ####
            answer_text = data["answer"]
            # Find the number after ####
            match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
            if match:
                ground_truth = match.group(1).replace(',', '')
            else:
                ground_truth = None
            
            examples.append({
                "question": data["question"],
                "answer": answer_text,
                "ground_truth": ground_truth,
            })
    
    return examples


def format_prompt(example: dict) -> str:
    """Format a GSM8K example into a prompt string."""
    return GSM8K_PROMPT_TEMPLATE.format(question=example["question"])


def evaluate_gsm8k(
    model_path: str,
    data_dir: Path,
    output_path: str = None,
    split: str = "test",
    max_examples: int = None,
) -> tuple[list[dict], float]:
    """
    Evaluate a model on GSM8K.
    
    Args:
        model_path: Path to the model
        data_dir: Path to GSM8K data directory
        output_path: Optional path to save results
        split: Which split to evaluate on
        max_examples: Maximum number of examples to evaluate (for testing)
    
    Returns:
        Tuple of (results list, accuracy percentage)
    """
    # Initialize model
    print(f"Loading model from {model_path}...")
    llm = LLM(model=model_path, dtype="bfloat16")
    
    # Sampling parameters - greedy decoding for evaluation
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,  # GSM8K needs more tokens for chain-of-thought
    )
    
    # Load data
    print(f"Loading GSM8K {split} data...")
    examples = load_gsm8k_data(data_dir, split)
    print(f"Loaded {len(examples)} examples from {split} split")
    
    if max_examples is not None:
        examples = examples[:max_examples]
        print(f"Using first {max_examples} examples")
    
    # Format prompts
    prompts = [format_prompt(ex) for ex in examples]
    
    # Generate responses
    print("Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_time = time.time() - start_time
    
    throughput = len(examples) / generation_time
    print(f"Generation completed in {generation_time:.2f}s ({throughput:.2f} examples/second)")
    
    # Grade responses
    results = []
    correct = 0
    failed_to_parse = 0
    
    print("Grading responses...")
    for example, output in tqdm(zip(examples, outputs), total=len(examples), desc="Grading"):
        response = output.outputs[0].text.strip()
        parsed = parse_gsm8k_response(response)
        
        # Compare numeric values
        is_correct = False
        if parsed is not None and example["ground_truth"] is not None:
            try:
                # Compare as floats to handle formatting differences
                is_correct = abs(float(parsed) - float(example["ground_truth"])) < 1e-6
            except ValueError:
                pass
        
        if is_correct:
            correct += 1
        
        if parsed is None:
            failed_to_parse += 1
        
        result = {
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "full_answer": example["answer"],
            "model_output": response,
            "parsed_response": parsed,
            "is_correct": is_correct,
        }
        results.append(result)
    
    # Calculate metrics
    total = len(results)
    accuracy = correct / total * 100
    
    print(f"\n{'='*60}")
    print(f"Results on {total} examples:")
    print(f"  Correct: {correct}/{total} ({accuracy:.2f}%)")
    print(f"  Failed to parse: {failed_to_parse} ({failed_to_parse/total*100:.2f}%)")
    print(f"  Throughput: {throughput:.2f} examples/second")
    print(f"{'='*60}")
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        summary_path = output_path.replace(".jsonl", "_summary.json")
        summary = {
            "model_path": model_path,
            "data_dir": str(data_dir),
            "split": split,
            "num_examples": total,
            "correct": correct,
            "accuracy": accuracy,
            "failed_to_parse": failed_to_parse,
            "throughput_examples_per_second": throughput,
            "generation_time_seconds": generation_time,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}")
        print(f"Summary saved to {summary_path}")
    
    return results, accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.1 8B on GSM8K")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to the model")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="Path to GSM8K data directory")
    parser.add_argument("--output_path", type=str, default="results/gsm8k_results.jsonl",
                        help="Path to save results")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples (for testing)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device to use (parsed early before imports)")
    
    args = parser.parse_args()
    
    results, accuracy = evaluate_gsm8k(
        model_path=args.model_path,
        data_dir=Path(args.data_dir),
        output_path=args.output_path,
        split=args.split,
        max_examples=args.max_examples,
    )
    
    print(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
