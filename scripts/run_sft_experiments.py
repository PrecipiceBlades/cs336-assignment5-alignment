#!/usr/bin/env python3
"""
Run SFT experiments on MATH dataset.

Experiment 1: Vary dataset sizes {128, 256, 512, 1024, full}
Experiment 2: Filter for correct answers only, run on filtered dataset
Experiment 3: Learning rate sweep on full dataset (1 epoch)
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# Data paths
SFT_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/train.jsonl"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
OUTPUT_DIR = Path("/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH")

# Add project to path
sys.path.insert(0, "/root/cs336-assignment5-alignment")


def filter_correct_examples():
    """Filter SFT examples to only include those with correct answers."""
    from cs336_alignment.drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn
    
    print("Filtering SFT examples for correct answers...")
    
    # Load SFT data and ground truths
    sft_data = []
    with open(SFT_DATA_PATH, 'r') as f:
        for line in f:
            sft_data.append(json.loads(line))
    
    train_data = []
    with open(TRAIN_DATA_PATH, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    if len(sft_data) != len(train_data):
        print(f"Warning: SFT data ({len(sft_data)}) and train data ({len(train_data)}) have different lengths!")
        print("Using minimum length...")
    
    # Filter correct examples
    correct_examples = []
    incorrect_count = 0
    
    for i, (sft_item, train_item) in enumerate(zip(sft_data, train_data)):
        response = sft_item["response"]
        ground_truth = extract_boxed_answer(train_item["solution"])
        
        # Add </answer> if not present (for grading)
        if not response.endswith("</answer>"):
            response_for_grading = response + "</answer>"
        else:
            response_for_grading = response
        
        reward_info = r1_zero_reward_fn(response_for_grading, ground_truth)
        
        if reward_info["answer_reward"] == 1.0:
            correct_examples.append(sft_item)
        else:
            incorrect_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(sft_data)}, correct so far: {len(correct_examples)}")
    
    print(f"\nFiltering complete:")
    print(f"  Total examples: {len(sft_data)}")
    print(f"  Correct examples: {len(correct_examples)}")
    print(f"  Incorrect examples: {incorrect_count}")
    print(f"  Accuracy: {len(correct_examples) / len(sft_data) * 100:.2f}%")
    
    # Save filtered data
    filtered_path = OUTPUT_DIR / "sft_correct_only.jsonl"
    with open(filtered_path, 'w') as f:
        for item in correct_examples:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nSaved filtered data to: {filtered_path}")
    return str(filtered_path), len(correct_examples)


def run_sft_experiment(
    num_samples: int,
    data_path: str,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 1,
    vllm_gpu: int = 0,
    vllm_gpu_memory_utilization: float = 0.3,
    wandb_project: str = "sft-math-experiments",
    experiment_name: str = None,
):
    """Run a single SFT experiment."""
    if experiment_name is None:
        experiment_name = f"sft-{num_samples}samples"
    
    cmd = [
        "uv", "run", "python", "-m", "cs336_alignment.sft",
        "--model_path", MODEL_PATH,
        "--train_data_path", data_path,
        "--val_data_path", VAL_DATA_PATH,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--num_epochs", str(num_epochs),
        "--vllm_gpu", str(vllm_gpu),
        "--vllm_gpu_memory_utilization", str(vllm_gpu_memory_utilization),
        "--wandb_project", wandb_project,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"  Data: {data_path}")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Grad accum steps: {gradient_accumulation_steps}")
    print(f"  Epochs: {num_epochs}")
    print(f"  GPU: {vllm_gpu}")
    print(f"{'='*60}\n")
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu)
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        cwd="/root/cs336-assignment5-alignment",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
    
    # Stream output
    accuracy = None
    for line in process.stdout:
        print(line, end='', flush=True)
        if "Final Validation Accuracy:" in line:
            try:
                accuracy = float(line.split(":")[-1].strip().replace("%", ""))
            except:
                pass
    
    process.wait()
    
    return {
        "experiment": experiment_name,
        "num_samples": num_samples,
        "learning_rate": learning_rate,
        "accuracy": accuracy,
        "return_code": process.returncode,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=["1", "2", "3", "all"], default="all",
                        help="Which experiment to run: 1 (dataset sizes), 2 (correct only), 3 (lr sweep), or all")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--vllm_gpu", type=int, default=0)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--wandb_project", type=str, default="sft-math-experiments")
    args = parser.parse_args()
    
    results = []
    
    # Experiment 1: Vary dataset sizes
    if args.experiment in ["1", "all"]:
        print("\n" + "="*80)
        print("EXPERIMENT 1: Varying dataset sizes")
        print("="*80)
        
        dataset_sizes = [128, 256, 512, 1024, 7500]  # Full dataset is 7500
        
        for num_samples in dataset_sizes:
            result = run_sft_experiment(
                num_samples=num_samples,
                data_path=SFT_DATA_PATH,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_epochs=args.num_epochs,
                vllm_gpu=args.vllm_gpu,
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                wandb_project=args.wandb_project,
                experiment_name=f"exp1-{num_samples}samples",
            )
            results.append(result)
            print(f"\nExperiment result: {result}")
    
    # Experiment 2: Correct answers only
    if args.experiment in ["2", "all"]:
        print("\n" + "="*80)
        print("EXPERIMENT 2: Correct answers only")
        print("="*80)
        
        # First, filter the data
        filtered_path, num_correct = filter_correct_examples()
        
        # Run SFT on filtered data
        result = run_sft_experiment(
            num_samples=num_correct,  # Use all correct examples
            data_path=filtered_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            vllm_gpu=args.vllm_gpu,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            wandb_project=args.wandb_project,
            experiment_name=f"exp2-correct-only-{num_correct}samples",
        )
        results.append(result)
        print(f"\nExperiment result: {result}")
    
    # Experiment 3: Learning rate sweep on full dataset (1 epoch)
    if args.experiment in ["3", "all"]:
        print("\n" + "="*80)
        print("EXPERIMENT 3: Learning rate sweep (full dataset, 1 epoch)")
        print("="*80)
        
        # Learning rates to sweep: from conservative to aggressive
        learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
        
        for lr in learning_rates:
            # Format LR nicely for experiment name
            lr_str = f"{lr:.0e}".replace("-0", "-")
            result = run_sft_experiment(
                num_samples=7500,  # Full dataset
                data_path=SFT_DATA_PATH,
                batch_size=args.batch_size,
                learning_rate=lr,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_epochs=1,  # Always 1 epoch for LR sweep
                vllm_gpu=args.vllm_gpu,
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                wandb_project=args.wandb_project,
                experiment_name=f"exp3-lr{lr_str}",
            )
            results.append(result)
            print(f"\nExperiment result: {result}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    for r in results:
        acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] is not None else "N/A"
        lr_str = f"{r['learning_rate']:.0e}".replace("-0", "-")
        print(f"  {r['experiment']}: {acc_str} (samples: {r['num_samples']}, lr: {lr_str})")
    
    # Save results to file
    results_path = OUTPUT_DIR / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
