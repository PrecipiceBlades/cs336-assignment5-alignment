"""Load the Anthropic HH dataset for DPO training."""
import gzip
import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class DPOExample:
    """A single DPO training example with instruction and chosen/rejected responses."""
    instruction: str
    chosen_response: str
    rejected_response: str
    source_file: str


def parse_conversation(text: str) -> Tuple[List[str], List[str]]:
    """Parse HH conversation format into human and assistant turns."""
    human_turns = []
    assistant_turns = []
    
    # Split by Human: and Assistant: markers
    parts = text.split("\n\nHuman: ")
    for i, part in enumerate(parts):
        if i == 0:
            continue  # Skip empty first part
        if "\n\nAssistant: " in part:
            human_msg, rest = part.split("\n\nAssistant: ", 1)
            human_turns.append(human_msg.strip())
            assistant_turns.append(rest.strip())
        else:
            human_turns.append(part.strip())
    
    return human_turns, assistant_turns


def load_anthropic_hh_dataset(data_dir: str) -> List[DPOExample]:
    """
    Load the Anthropic HH dataset for DPO training.
    
    Combines all 4 files (harmless-base, helpful-base, helpful-online, 
    helpful-rejection-sampled), filters to single-turn conversations,
    and extracts instruction + chosen/rejected responses.
    
    Args:
        data_dir: Path to directory containing the .jsonl.gz files
        
    Returns:
        List of DPOExample objects ready for training
    """
    data_path = Path(data_dir)
    files = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz", 
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz"
    ]
    
    examples = []
    
    for filename in files:
        filepath = data_path / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping")
            continue
            
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                chosen = data["chosen"]
                rejected = data["rejected"]
                
                # Parse conversations
                chosen_humans, chosen_assistants = parse_conversation(chosen)
                rejected_humans, rejected_assistants = parse_conversation(rejected)
                
                # Filter: only keep single-turn (one human message)
                if len(chosen_humans) != 1 or len(rejected_humans) != 1:
                    continue
                
                # Verify same instruction for chosen and rejected
                if chosen_humans[0] != rejected_humans[0]:
                    continue
                    
                if not chosen_assistants or not rejected_assistants:
                    continue
                
                examples.append(DPOExample(
                    instruction=chosen_humans[0],
                    chosen_response=chosen_assistants[0],
                    rejected_response=rejected_assistants[0],
                    source_file=filename.replace(".jsonl.gz", "")
                ))
    
    return examples


if __name__ == "__main__":
    # Example usage
    DATA_DIR = "/mnt/fsx/fsx-pred/prediction/tdvi/data/a5-alignment/hh"
    examples = load_anthropic_hh_dataset(DATA_DIR)
    print(f"Loaded {len(examples)} DPO examples")
    
    # Count by source
    from collections import Counter
    sources = Counter(e.source_file for e in examples)
    for source, count in sources.items():
        print(f"  {source}: {count}")
