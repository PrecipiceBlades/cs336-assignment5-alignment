"""Debug SFT model outputs to understand low validation accuracy."""
import os
import json
from pathlib import Path

# Set GPU before imports
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer

MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
PROMPT_TEMPLATE = (Path(__file__).parent / "cs336_alignment/prompts/r1_zero.prompt").read_text()

def load_val_data(data_path: str, n=5):
    """Load first n validation examples."""
    prompts = []
    ground_truths = []
    raw_solutions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            question = data["problem"]
            prompt = PROMPT_TEMPLATE.format(question=question)
            prompts.append(prompt)
            ground_truths.append(extract_boxed_answer(data["solution"]))
            raw_solutions.append(data["solution"][:200])
    return prompts, ground_truths, raw_solutions

def main():
    print("Loading validation data...")
    val_prompts, val_ground_truths, raw_solutions = load_val_data(VAL_DATA_PATH, n=5)
    
    print(f"\n{'='*60}")
    print("PROMPT TEMPLATE (showing end):")
    print(repr(PROMPT_TEMPLATE[-100:]))
    print(f"{'='*60}\n")
    
    print("Initializing vLLM with BASE model...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        seed=42,
    )
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=512,  # shorter for debugging
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    print("\nGenerating responses...")
    outputs = llm.generate(val_prompts, sampling_params)
    
    for i, (output, gt, raw_sol) in enumerate(zip(outputs, val_ground_truths, raw_solutions)):
        response = output.outputs[0].text
        if not response.endswith("</answer>"):
            response = response + "</answer>"
        
        reward_info = r1_zero_reward_fn(response, gt)
        
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*60}")
        print(f"Ground Truth Answer: {gt}")
        print(f"Raw Solution Start: {raw_sol}...")
        print(f"\nPROMPT END: {repr(val_prompts[i][-80:])}")
        print(f"\nGENERATED RESPONSE (first 500 chars):")
        print("-" * 40)
        print(response[:500])
        print("-" * 40)
        print(f"\nRESPONSE END (last 200 chars):")
        print(repr(response[-200:]))
        print(f"\nReward info: {reward_info}")
        print(f"Has '</think> <answer>': {'</think> <answer>' in response}")

if __name__ == "__main__":
    main()
