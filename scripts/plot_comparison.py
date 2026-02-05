"""Compare validation accuracy across multiple GRPO experiments."""

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


def plot_comparison(experiments: dict, output_path: str, title: str = "GRPO Experiments Comparison"):
    """Create and save comparison plot.
    
    Args:
        experiments: dict mapping experiment name to (steps, accuracies) tuple
        output_path: path to save the plot
        title: plot title
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (name, (steps, accuracies)) in enumerate(experiments.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(steps, accuracies, f'-{marker}', color=color, linewidth=2, 
                markersize=8, label=name, alpha=0.8)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare GRPO experiments")
    parser.add_argument("--logs", nargs="+", required=True, help="Log file paths")
    parser.add_argument("--names", nargs="+", required=True, help="Experiment names")
    parser.add_argument("--output", default="grpo_comparison.png", help="Output path")
    parser.add_argument("--title", default="GRPO Experiments Comparison", help="Plot title")
    args = parser.parse_args()
    
    if len(args.logs) != len(args.names):
        raise ValueError("Number of logs must match number of names")
    
    experiments = {}
    for log_path, name in zip(args.logs, args.names):
        steps, accuracies = parse_log_file(log_path)
        if steps:
            experiments[name] = (steps, accuracies)
            print(f"{name}: {len(steps)} data points, final accuracy: {accuracies[-1]:.2f}%")
        else:
            print(f"Warning: No data found in {log_path}")
    
    if experiments:
        plot_comparison(experiments, args.output, args.title)
    else:
        print("No valid experiment data found.")


if __name__ == "__main__":
    main()
