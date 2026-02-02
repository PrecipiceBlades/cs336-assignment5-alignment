"""Generate example rollouts from the trained GRPO model."""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Paths
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
PROMPT_TEMPLATE = (Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt").read_text()

def load_val_data(n_samples=3):
    """Load a few validation examples."""
    prompts = []
    ground_truths = []
    problems = []
    with open(VAL_DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data = json.loads(line)
            question = data.get("question") or data.get("problem")
            prompt = PROMPT_TEMPLATE.format(question=question)
            prompts.append(prompt)
            problems.append(question)
            # Extract ground truth answer
            solution = data["solution"]
            # Try to find boxed answer
            if "\\boxed{" in solution:
                import re
                match = re.search(r'\\boxed\{([^}]+)\}', solution)
                if match:
                    ground_truths.append(match.group(1))
                else:
                    ground_truths.append(solution[-100:])
            else:
                ground_truths.append(solution[-100:])
    return problems, prompts, ground_truths


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    print("Loading validation examples...")
    problems, prompts, ground_truths = load_val_data(3)
    
    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    print("\nGenerating rollouts...")
    outputs = llm.generate(prompts, sampling_params)
    
    print("\n" + "="*80)
    print("EXAMPLE ROLLOUTS FROM BASE MODEL")
    print("="*80)
    
    for i, (problem, output, gt) in enumerate(zip(problems, outputs, ground_truths)):
        print(f"\n--- Example {i+1} ---")
        print(f"Problem: {problem[:200]}...")
        print(f"\nGround Truth Answer: {gt}")
        print(f"\nModel Response:")
        response = output.outputs[0].text
        # Show first 500 chars of response
        if len(response) > 500:
            print(response[:500] + "...[truncated]")
        else:
            print(response)
        print()


if __name__ == "__main__":
    main()
