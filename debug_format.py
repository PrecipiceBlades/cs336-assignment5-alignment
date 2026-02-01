"""Debug the format mismatch between training and validation."""
import json
from pathlib import Path

# Import from sft.py to use the SAME logic
from cs336_alignment.sft import load_train_data, load_val_data, PROMPT_TEMPLATE

TRAIN_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/sft.jsonl"
VAL_DATA_PATH = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/MATH/validation.jsonl"

def main():
    print("=" * 70)
    print("PROMPT TEMPLATE:")
    print("=" * 70)
    print(repr(PROMPT_TEMPLATE))
    print()
    
    print("=" * 70)
    print("TRAINING FORMAT (from sft.py load_train_data):")
    print("=" * 70)
    
    train_prompts, train_responses = load_train_data(TRAIN_DATA_PATH, num_samples=1)
    prompt = train_prompts[0]
    response = train_responses[0]
    
    print("Training Prompt (last 100 chars):")
    print(repr(prompt[-100:]))
    print()
    print("Training Response (first 100 chars):")
    print(repr(response[:100]))
    print()
    
    print("=" * 70)
    print("VALIDATION FORMAT (from sft.py load_val_data):")
    print("=" * 70)
    
    val_prompts, val_ground_truths = load_val_data(VAL_DATA_PATH)
    val_prompt = val_prompts[0]
    
    print("Validation Prompt (last 100 chars):")
    print(repr(val_prompt[-100:]))
    print()
    
    print("=" * 70)
    print("FORMAT CHECK:")
    print("=" * 70)
    print("Training prompt ends with:   ", repr(prompt[-30:]))
    print("Validation prompt ends with: ", repr(val_prompt[-30:]))
    
    if prompt[-30:].endswith("<think>") and val_prompt[-30:].endswith("<think>"):
        print("\n[OK] Both prompts end with '<think>' - formats match!")
    else:
        print("\n[ERROR] Format mismatch detected!")

if __name__ == "__main__":
    main()
