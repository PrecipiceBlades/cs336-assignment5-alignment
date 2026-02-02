"""
Evaluate a vLLM model on the MATH validation set.

This script:
1. Loads the MATH validation data
2. Extracts ground truth answers from \boxed{} in solutions
3. Generates responses using vLLM
4. Grades responses using r1_zero_reward_fn
5. Reports accuracy metrics by category
"""
import json
import os
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from drgrpo_grader import extract_boxed_answer, r1_zero_reward_fn

# Paths
VALIDATION_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"
MODEL_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/models/Qwen2.5-Math-1.5B"

# Load prompt template
PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "r1_zero.prompt").read_text()

# Sampling parameters
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True
)


def evaluate_vllm(data_path: str, model_path: str, output_path: str = None) -> list[dict]:
    """
    Evaluate a model on the MATH validation set.
    
    Args:
        data_path: Path to validation JSONL file
        model_path: Path to the model
        output_path: Optional path to save results
        
    Returns:
        List of result dicts and accuracy metrics
    """
    # Initialize model
    llm = LLM(model=model_path)
    
    # Load data
    prompts = []
    questions = []
    ground_truths = []
    
    print("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            data = json.loads(line)
            question = data["problem"]
            questions.append(question)
            prompt = PROMPT_TEMPLATE.format(question=question)
            prompts.append(prompt)
            ground_truths.append(extract_boxed_answer(data["solution"]))

    # Generate responses
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Grade responses
    results = []
    cat1_format1_answer1 = 0
    cat2_format1_answer0 = 0
    cat3_format0_answer0 = 0
    
    print("Grading responses...")
    for i, (question, ground_truth, output) in enumerate(tqdm(
        zip(questions, ground_truths, outputs), 
        desc="Grading responses", 
        total=len(questions)
    )):
        response = output.outputs[0].text
        if not response.endswith("</answer>"):
            response = response + "</answer>"
        
        reward_info = r1_zero_reward_fn(response, ground_truth)
        
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            **reward_info,
        }
        results.append(result)
        
        fmt = reward_info["format_reward"]
        ans = reward_info["answer_reward"]
        if fmt == 1.0 and ans == 1.0:
            cat1_format1_answer1 += 1
        elif fmt == 1.0 and ans == 0.0:
            cat2_format1_answer0 += 1
        elif fmt == 0.0 and ans == 0.0:
            cat3_format0_answer0 += 1
    
    # Calculate metrics
    total = len(results)
    accuracy = cat1_format1_answer1 / total * 100
    format_accuracy = (cat1_format1_answer1 + cat2_format1_answer0) / total * 100
    
    print(f"\n{'='*60}")
    print(f"Results on {total} examples:")
    print(f"  Category 1 (format=1, answer=1): {cat1_format1_answer1} ({cat1_format1_answer1/total*100:.2f}%)")
    print(f"  Category 2 (format=1, answer=0): {cat2_format1_answer0} ({cat2_format1_answer0/total*100:.2f}%)")
    print(f"  Category 3 (format=0, answer=0): {cat3_format0_answer0} ({cat3_format0_answer0/total*100:.2f}%)")
    print(f"  Answer Accuracy: {cat1_format1_answer1}/{total} ({accuracy:.2f}%)")
    print(f"  Format Accuracy: {cat1_format1_answer1 + cat2_format1_answer0}/{total} ({format_accuracy:.2f}%)")
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
            "data_path": data_path,
            "num_examples": total,
            "cat1_format1_answer1": cat1_format1_answer1,
            "cat2_format1_answer0": cat2_format1_answer0,
            "cat3_format0_answer0": cat3_format0_answer0,
            "accuracy": accuracy,
            "format_accuracy": format_accuracy,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {output_path}")
        print(f"Summary saved to {summary_path}")
    
    return results, accuracy, format_accuracy


def main():
    results, accuracy, format_accuracy = evaluate_vllm(
        VALIDATION_DATA_PATH, 
        MODEL_PATH, 
        "results/vllm_results.jsonl"
    )
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Format Accuracy: {format_accuracy:.2f}%")


if __name__ == "__main__":
    main()
