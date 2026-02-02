"""
Evaluate Llama 3.1 8B zero-shot performance on MMLU.

This script:
1. Loads the MMLU test data from CSV files
2. Formats them as prompts to the language model
3. Generates outputs for each example using vLLM
4. Parses model outputs to extract predicted answers
5. Calculates evaluation metrics (accuracy)
6. Serializes examples, model generations, and evaluation scores to disk

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python -m cs336_alignment.evaluate_mmlu_zero_shot
"""
import csv
import json
import os
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

from .utils import parse_mmlu_response

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "mmlu"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Llama-3.1-8B"

# Load system prompt template
SYSTEM_PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "zero_shot_system_prompt.prompt").read_text()

# MMLU instruction template (arrows in assignment indicate line continuation, not newlines)
MMLU_INSTRUCTION_TEMPLATE = """Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer:"""


def load_mmlu_data(data_dir: Path, split: str = "test") -> list[dict]:
    """
    Load MMLU data from CSV files.
    
    Args:
        data_dir: Path to MMLU data directory
        split: Which split to load ("dev", "val", or "test")
    
    Returns:
        List of MMLU examples with keys: subject, question, options, answer
    """
    split_dir = data_dir / split
    examples = []
    
    for csv_file in sorted(split_dir.glob("*.csv")):
        # Extract subject from filename (e.g., "abstract_algebra_test.csv" -> "abstract_algebra")
        subject = csv_file.stem.replace(f"_{split}", "")
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    question, opt_a, opt_b, opt_c, opt_d, answer = row[:6]
                    examples.append({
                        "subject": subject,
                        "question": question,
                        "options": [opt_a, opt_b, opt_c, opt_d],
                        "answer": answer.strip().upper(),
                    })
    
    return examples


def format_prompt(example: dict) -> str:
    """Format an MMLU example into a prompt string using the system prompt template."""
    # First create the MMLU instruction
    instruction = MMLU_INSTRUCTION_TEMPLATE.format(
        subject=example["subject"].replace("_", " "),
        question=example["question"],
        option_a=example["options"][0],
        option_b=example["options"][1],
        option_c=example["options"][2],
        option_d=example["options"][3],
    )
    # Then insert into the system prompt template
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction)


def evaluate_mmlu(
    model_path: str,
    data_dir: Path,
    output_path: str = None,
    split: str = "test",
    max_examples: int = None,
) -> tuple[list[dict], float]:
    """
    Evaluate a model on MMLU.
    
    Args:
        model_path: Path to the model
        data_dir: Path to MMLU data directory
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
        max_tokens=256,
    )
    
    # Load data
    print(f"Loading MMLU {split} data...")
    examples = load_mmlu_data(data_dir, split)
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
    
    # Track per-subject accuracy
    subject_correct = {}
    subject_total = {}
    
    print("Grading responses...")
    for example, output in tqdm(zip(examples, outputs), total=len(examples), desc="Grading"):
        response = output.outputs[0].text.strip()
        parsed = parse_mmlu_response(example, response)
        
        is_correct = (parsed == example["answer"])
        if is_correct:
            correct += 1
        
        if parsed is None:
            failed_to_parse += 1
        
        # Track per-subject
        subject = example["subject"]
        subject_correct[subject] = subject_correct.get(subject, 0) + (1 if is_correct else 0)
        subject_total[subject] = subject_total.get(subject, 0) + 1
        
        result = {
            "subject": subject,
            "question": example["question"],
            "options": example["options"],
            "ground_truth": example["answer"],
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
    
    # Print per-subject accuracy (sorted by accuracy)
    print("\nPer-subject accuracy:")
    subject_accuracies = {
        subj: subject_correct[subj] / subject_total[subj] * 100
        for subj in subject_total
    }
    for subj in sorted(subject_accuracies, key=subject_accuracies.get, reverse=True)[:10]:
        print(f"  {subj}: {subject_accuracies[subj]:.2f}% ({subject_correct[subj]}/{subject_total[subj]})")
    print("  ...")
    
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
            "subject_accuracies": subject_accuracies,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}")
        print(f"Summary saved to {summary_path}")
    
    return results, accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.1 8B on MMLU")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to the model")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="Path to MMLU data directory")
    parser.add_argument("--output_path", type=str, default="results/mmlu_results.jsonl",
                        help="Path to save results")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples (for testing)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device to use (parsed early before imports)")
    
    args = parser.parse_args()
    
    results, accuracy = evaluate_mmlu(
        model_path=args.model_path,
        data_dir=Path(args.data_dir),
        output_path=args.output_path,
        split=args.split,
        max_examples=args.max_examples,
    )
    
    print(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
