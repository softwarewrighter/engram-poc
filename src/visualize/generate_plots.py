"""Generate visualization plots for Engram PoC results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    """Configure matplotlib for light academic style."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.color': '#cccccc',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi': 150,
    })


def generate_mlx_loss_curve(output_path: Path):
    """Generate loss curve from MLX training data."""
    # Data from results/training_run_001.md
    iterations = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_loss = [None, 4.140, 3.327, 2.745, 2.421, 2.367, 2.174, 2.070, 1.891, 1.743, 1.755]
    val_loss = [4.344, None, None, None, None, 2.365, None, None, None, None, 1.815]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot training loss (skip None values)
    train_iters = [i for i, l in zip(iterations, train_loss) if l is not None]
    train_vals = [l for l in train_loss if l is not None]
    ax.plot(train_iters, train_vals, 'o-', color='#2563eb', linewidth=2,
            markersize=6, label='Training Loss')

    # Plot validation loss
    val_iters = [i for i, l in zip(iterations, val_loss) if l is not None]
    val_vals = [l for l in val_loss if l is not None]
    ax.plot(val_iters, val_vals, 's--', color='#dc2626', linewidth=2,
            markersize=8, label='Validation Loss')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('MLX Training Loss Progression')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_xlim(0, 105)
    ax.set_ylim(1.5, 4.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def generate_mlx_accuracy_comparison(output_path: Path):
    """Generate accuracy comparison bar chart for MLX."""
    # Data from comparison.json
    labels = ['Baseline', 'Engram-tuned']
    accuracies = [8.65, 11.54]
    colors = ['#94a3b8', '#2563eb']

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(labels, accuracies, color=colors, edgecolor='#333333', linewidth=1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MLX Accuracy: Baseline vs Engram-tuned')
    ax.set_ylim(0, 15)
    ax.grid(True, axis='y')

    # Add improvement annotation
    ax.annotate('+33.3% relative',
                xy=(1, 11.54), xytext=(1.3, 13),
                arrowprops=dict(arrowstyle='->', color='#16a34a'),
                fontsize=10, color='#16a34a', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def generate_mlx_category_performance(output_path: Path, comparison_data: dict):
    """Generate horizontal bar chart showing per-category accuracy changes."""
    baseline_cats = comparison_data['baseline']['by_category']
    tuned_cats = comparison_data['tuned']['by_category']

    # Get categories with changes
    categories = []
    baseline_acc = []
    tuned_acc = []

    for cat in baseline_cats:
        if cat in tuned_cats:
            b_acc = baseline_cats[cat]['accuracy'] * 100
            t_acc = tuned_cats[cat]['accuracy'] * 100
            categories.append(cat.replace('_', ' ').title())
            baseline_acc.append(b_acc)
            tuned_acc.append(t_acc)

    # Sort by improvement
    improvements = [t - b for b, t in zip(baseline_acc, tuned_acc)]
    sorted_indices = np.argsort(improvements)[::-1]

    # Take top 12 categories for readability
    top_n = min(12, len(categories))
    sorted_indices = sorted_indices[:top_n]

    categories = [categories[i] for i in sorted_indices]
    baseline_acc = [baseline_acc[i] for i in sorted_indices]
    tuned_acc = [tuned_acc[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(categories))
    height = 0.35

    bars1 = ax.barh(y - height/2, baseline_acc, height, label='Baseline',
                    color='#94a3b8', edgecolor='#333333')
    bars2 = ax.barh(y + height/2, tuned_acc, height, label='Engram-tuned',
                    color='#2563eb', edgecolor='#333333')

    ax.set_xlabel('Accuracy (%)')
    ax.set_title('MLX Per-Category Accuracy Comparison')
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.legend(loc='lower right')
    ax.grid(True, axis='x')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def generate_cuda_accuracy_comparison(output_path: Path):
    """Generate accuracy comparison bar chart for CUDA/Unsloth."""
    # Data from unsloth-nvidia/results/comparison.json
    labels = ['Baseline', 'Engram-tuned']
    accuracies = [8.59, 14.06]
    colors = ['#94a3b8', '#16a34a']

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(labels, accuracies, color=colors, edgecolor='#333333', linewidth=1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CUDA/Unsloth Accuracy: Baseline vs Engram-tuned')
    ax.set_ylim(0, 18)
    ax.grid(True, axis='y')

    # Add improvement annotation
    ax.annotate('+63.6% relative',
                xy=(1, 14.06), xytext=(1.3, 16),
                arrowprops=dict(arrowstyle='->', color='#16a34a'),
                fontsize=10, color='#16a34a', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def main():
    """Generate all plots."""
    setup_style()

    project_root = Path(__file__).parent.parent.parent
    plots_dir = project_root / 'images' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load MLX comparison data
    comparison_path = project_root / 'results' / 'comparison.json'
    with open(comparison_path) as f:
        comparison_data = json.load(f)

    print("Generating plots...")

    # MLX plots
    generate_mlx_loss_curve(plots_dir / 'mlx-loss-curve.png')
    generate_mlx_accuracy_comparison(plots_dir / 'mlx-accuracy-comparison.png')
    generate_mlx_category_performance(plots_dir / 'mlx-category-performance.png', comparison_data)

    # CUDA plot
    generate_cuda_accuracy_comparison(plots_dir / 'cuda-accuracy-comparison.png')

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()
