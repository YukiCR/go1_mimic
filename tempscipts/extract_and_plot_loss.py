#!/usr/bin/env python3
"""
Extract loss values from robomimic training log and plot training curves.
Also provides recommendations on whether to continue training.
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """Parse the log file to extract training metrics."""

    epochs = []
    losses = []
    time_epochs = []
    time_train_batches = []

    with open(log_path, 'r') as f:
        content = f.read()

    # Split by "Train Epoch" to get per-epoch data
    epoch_blocks = re.split(r'Train Epoch (\d+)', content)

    # epoch_blocks[0] is content before first epoch
    # epoch_blocks[1], epoch_blocks[2]... are (epoch_num, epoch_content) pairs

    for i in range(1, len(epoch_blocks), 2):
        if i + 1 < len(epoch_blocks):
            epoch_num = int(epoch_blocks[i])
            epoch_content = epoch_blocks[i + 1]

            # Extract Loss
            loss_match = re.search(r'"Loss":\s*([-\d.]+)', epoch_content)
            # Extract Time_Epoch
            time_epoch_match = re.search(r'"Time_Epoch":\s*([\d.]+)', epoch_content)
            # Extract Time_Train_Batch
            time_batch_match = re.search(r'"Time_Train_Batch":\s*([\d.]+)', epoch_content)

            if loss_match:
                epochs.append(epoch_num)
                losses.append(float(loss_match.group(1)))

                if time_epoch_match:
                    time_epochs.append(float(time_epoch_match.group(1)))
                else:
                    time_epochs.append(None)

                if time_batch_match:
                    time_train_batches.append(float(time_batch_match.group(1)))
                else:
                    time_train_batches.append(None)

    return {
        'epochs': epochs,
        'losses': losses,
        'time_epochs': time_epochs,
        'time_train_batches': time_train_batches
    }


def plot_training_curve(data, output_path):
    """Plot the training loss curve."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = data['epochs']
    losses = data['losses']

    # Plot 1: Full training curve
    ax1 = axes[0]
    ax1.plot(epochs, losses, 'b-', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (Full)')
    ax1.grid(True, alpha=0.3)

    # Add smoothed curve
    if len(losses) >= 10:
        window = min(20, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(epochs[window-1:], smoothed, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
        ax1.legend()

    # Plot 2: Recent training curve (last 100 epochs)
    ax2 = axes[1]
    recent_epochs = epochs[-100:] if len(epochs) > 100 else epochs
    recent_losses = losses[-100:] if len(losses) > 100 else losses

    ax2.plot(recent_epochs, recent_losses, 'b-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Last 100 Epochs)' if len(epochs) > 100 else 'Training Loss (All Epochs)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    return fig


def analyze_training(data):
    """Analyze training progress and provide recommendations."""

    losses = data['losses']
    epochs = data['epochs']

    if len(losses) < 2:
        return "Insufficient data for analysis."

    analysis = []
    analysis.append("=" * 60)
    analysis.append("TRAINING ANALYSIS REPORT")
    analysis.append("=" * 60)
    analysis.append(f"Total Epochs Trained: {len(epochs)}")
    analysis.append(f"Current Epoch: {epochs[-1]}")
    analysis.append("")

    # Loss statistics
    analysis.append("LOSS STATISTICS:")
    analysis.append(f"  Initial Loss: {losses[0]:.4f} (Epoch {epochs[0]})")
    analysis.append(f"  Current Loss: {losses[-1]:.4f} (Epoch {epochs[-1]})")
    analysis.append(f"  Best Loss: {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})")
    analysis.append(f"  Worst Loss: {max(losses):.4f} (Epoch {epochs[losses.index(max(losses))]})")
    analysis.append(f"  Mean Loss: {np.mean(losses):.4f}")
    analysis.append(f"  Std Dev: {np.std(losses):.4f}")
    analysis.append("")

    # Trend analysis
    # Overall trend
    first_half = losses[:len(losses)//2]
    second_half = losses[len(losses)//2:]

    analysis.append("TREND ANALYSIS:")
    analysis.append(f"  First Half Avg Loss: {np.mean(first_half):.4f}")
    analysis.append(f"  Second Half Avg Loss: {np.mean(second_half):.4f}")

    # Recent trend (last 50 epochs)
    if len(losses) >= 50:
        recent = losses[-50:]
        earlier = losses[-100:-50] if len(losses) >= 100 else losses[:len(losses)//2]

        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)

        analysis.append(f"  Last 50 Epochs Avg: {recent_mean:.4f}")
        if len(losses) >= 100:
            analysis.append(f"  Previous 50 Epochs Avg: {earlier_mean:.4f}")

        # Calculate improvement rate
        if len(losses) >= 10:
            recent_slope = (losses[-1] - losses[-10]) / 10
            analysis.append(f"  Recent Slope (last 10 epochs): {recent_slope:.4f} loss/epoch")

    analysis.append("")

    # Convergence analysis
    if len(losses) >= 50:
        recent_losses = losses[-50:]
        variance = np.var(recent_losses)
        analysis.append(f"  Recent Variance (last 50 epochs): {variance:.4f}")

    analysis.append("")
    analysis.append("=" * 60)
    analysis.append("RECOMMENDATION:")
    analysis.append("=" * 60)

    # Generate recommendation
    recommendation = generate_recommendation(losses, epochs)
    analysis.append(recommendation)
    analysis.append("")

    return "\n".join(analysis)


def generate_recommendation(losses, epochs):
    """Generate training recommendation based on loss curve."""

    if len(losses) < 10:
        return "TRAINING TOO SHORT\n" + \
               "  - Train for at least 50-100 epochs before evaluating.\n" + \
               "  - Continue training to see meaningful trends."

    # Check for convergence
    recent_losses = losses[-min(50, len(losses)//4):]
    recent_variance = np.var(recent_losses)
    recent_mean = np.mean(recent_losses)

    # Check overall trend
    overall_improvement = losses[0] - losses[-1]

    # Check recent trend (last 20 epochs)
    if len(losses) >= 30:
        recent_trend = losses[-1] - losses[-20]
    else:
        recent_trend = losses[-1] - losses[0]

    recommendations = []

    # Determine if loss is decreasing
    is_improving = recent_trend < -0.1  # Loss is decreasing
    is_converged = recent_variance < 0.5 and len(losses) > 100  # Low variance after sufficient training
    is_diverging = recent_trend > 0.5  # Loss is increasing significantly
    is_plateau = abs(recent_trend) < 0.1 and len(losses) > 100  # Flat curve

    if is_diverging:
        recommendations.append("STOP TRAINING - LOSS INCREASING")
        recommendations.append("  - The loss is increasing, which may indicate overfitting.")
        recommendations.append("  - Consider:")
        recommendations.append("    * Reducing learning rate")
        recommendations.append("    * Adding regularization")
        recommendations.append("    * Stopping and using an earlier checkpoint")

    elif is_converged:
        recommendations.append("CONSIDER STOPPING - TRAINING CONVERGED")
        recommendations.append("  - The loss has stabilized with low variance.")
        recommendations.append("  - Further training may not provide significant benefits.")
        recommendations.append("  - Recommendation: Stop training and evaluate the model.")

    elif is_plateau:
        recommendations.append("PLATEAU DETECTED - CONSIDER ADJUSTMENTS")
        recommendations.append("  - Training progress has slowed significantly.")
        recommendations.append("  - Options:")
        recommendations.append("    * Continue for a bit longer to confirm plateau")
        recommendations.append("    * Reduce learning rate (if using adaptive LR)")
        recommendations.append("    * Stop if validation performance is satisfactory")

    elif is_improving:
        recommendations.append("CONTINUE TRAINING - STILL IMPROVING")
        recommendations.append(f"  - Loss decreased by {abs(recent_trend):.4f} in recent epochs.")
        recommendations.append("  - Model is still learning. Continue training.")

        # Estimate remaining training needed
        if overall_improvement > 0:
            avg_improvement_per_epoch = overall_improvement / len(losses)
            if avg_improvement_per_epoch > 0.01:
                recommendations.append(f"  - Current improvement rate: ~{avg_improvement_per_epoch:.4f} loss/epoch")
                recommendations.append("  - Suggested: Continue for at least 50-100 more epochs.")

    else:
        recommendations.append("MONITOR TRAINING - UNCLEAR TREND")
        recommendations.append("  - The loss trend is not clearly improving or worsening.")
        recommendations.append("  - Continue training and monitor for a clear pattern.")

    # Additional metrics context
    recommendations.append("")
    recommendations.append("KEY OBSERVATIONS:")
    if overall_improvement > 0:
        recommendations.append(f"  - Overall improvement: {overall_improvement:.4f} ({100*overall_improvement/abs(losses[0]):.1f}% reduction)")
    else:
        recommendations.append(f"  - Overall change: {overall_improvement:.4f}")

    recommendations.append(f"  - Recent loss variance: {recent_variance:.4f} (lower is more stable)")

    return "\n".join(recommendations)


def main():
    parser = argparse.ArgumentParser(
        description="Extract loss values from robomimic training log and plot training curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_image_go1_nav_lader/20260312222104/logs/log.txt",
        help="Path to the training log file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/tempoutputs",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--plot-filename",
        type=str,
        default="training_loss_curve.png",
        help="Filename for the training curve plot"
    )
    parser.add_argument(
        "--report-filename",
        type=str,
        default="training_analysis_report.txt",
        help="Filename for the analysis report"
    )

    args = parser.parse_args()

    # Construct full paths
    log_path = args.log_path
    output_dir = args.output_dir
    plot_path = os.path.join(output_dir, args.plot_filename)
    report_path = os.path.join(output_dir, args.report_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)

    print(f"Found {len(data['epochs'])} epochs of training data")

    # Plot training curve
    print("\nGenerating training curve plot...")
    plot_training_curve(data, plot_path)

    # Analyze training and generate recommendations
    print("\nAnalyzing training progress...")
    analysis = analyze_training(data)
    print(analysis)

    # Save report
    with open(report_path, 'w') as f:
        f.write(analysis)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
