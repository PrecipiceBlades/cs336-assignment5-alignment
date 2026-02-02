"""Generate GRPO training report: plot validation accuracy and save example rollouts."""

import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_path: str):
    """Parse GRPO training log to extract validation accuracies."""
    steps = []
    accuracies = []
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Step (\d+): Val Accuracy = ([\d.]+)%', line)
            if match:
                step = int(match.group(1))
                accuracy = float(match.group(2))
                steps.append(step)
                accuracies.append(accuracy)
    
    return steps, accuracies


def plot_validation_accuracy(steps, accuracies, output_path: str):
    """Create and save validation accuracy plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('GRPO Training: Validation Accuracy over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for step, acc in zip(steps, accuracies):
        plt.annotate(f'{acc:.1f}%', (step, acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    log_path = "/root/cs336-assignment5-alignment/grpo_training.log"
    output_dir = Path("/root/cs336-assignment5-alignment/scripts")
    
    steps, accuracies = parse_log_file(log_path)
    
    if not steps:
        print("No validation accuracy data found in log file.")
        return
    
    print("Validation Accuracy Results:")
    print("-" * 40)
    for step, acc in zip(steps, accuracies):
        print(f"Step {step:3d}: {acc:.2f}%")
    print("-" * 40)
    
    if len(accuracies) >= 2:
        improvement = accuracies[-1] - accuracies[0]
        print(f"Total improvement: +{improvement:.2f}%")
    
    plot_path = output_dir / "grpo_validation_accuracy.png"
    plot_validation_accuracy(steps, accuracies, str(plot_path))


if __name__ == "__main__":
    main()
