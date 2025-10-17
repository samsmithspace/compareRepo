"""
Script to compare training times between Traditional and ALaST fine-tuning
Run this after training both models
"""

import json
import os
from datetime import timedelta


def format_time(seconds):
    """Format seconds into a readable string"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compare_training_times(traditional_path='results/traditional_beans_metrics.json',
                           alast_path='results/alast_beans_metrics.json'):
    """Compare training times between traditional and ALaST"""

    print("=" * 80)
    print("TRAINING TIME COMPARISON: Traditional vs ALaST")
    print("=" * 80)

    # Check if files exist
    if not os.path.exists(traditional_path):
        print(f"\nError: Traditional metrics not found at {traditional_path}")
        print("Please run traditional training first.")
        return

    if not os.path.exists(alast_path):
        print(f"\nError: ALaST metrics not found at {alast_path}")
        print("Please run ALaST training first.")
        return

    # Load metrics
    with open(traditional_path, 'r') as f:
        trad_metrics = json.load(f)

    with open(alast_path, 'r') as f:
        alast_metrics = json.load(f)

    # Extract timing information
    trad_total = trad_metrics.get('total_training_time')
    alast_total = alast_metrics.get('total_training_time')

    # If total time not available, calculate from epoch times
    if trad_total is None:
        trad_total = sum(trad_metrics['time_per_epoch'])
        print("\n(Note: Traditional total time calculated from epoch times)")

    if alast_total is None:
        alast_total = sum(alast_metrics['time_per_epoch'])
        print("\n(Note: ALaST total time calculated from epoch times)")

    # Calculate statistics
    trad_avg_epoch = sum(trad_metrics['time_per_epoch']) / len(trad_metrics['time_per_epoch'])
    alast_avg_epoch = sum(alast_metrics['time_per_epoch']) / len(alast_metrics['time_per_epoch'])

    time_saved = trad_total - alast_total
    speedup = trad_total / alast_total if alast_total > 0 else 0
    percent_faster = (time_saved / trad_total * 100) if trad_total > 0 else 0

    # Get accuracy information
    trad_best_acc = max(trad_metrics['val_acc'])
    alast_best_acc = max(alast_metrics['val_acc'])
    acc_diff = alast_best_acc - trad_best_acc

    # Print comparison
    print("\n" + "-" * 80)
    print("TOTAL TRAINING TIME")
    print("-" * 80)
    print(f"{'Method':<20} {'Time (seconds)':<20} {'Time (formatted)':<20}")
    print("-" * 80)
    print(f"{'Traditional':<20} {trad_total:<20.2f} {format_time(trad_total):<20}")
    print(f"{'ALaST':<20} {alast_total:<20.2f} {format_time(alast_total):<20}")
    print("-" * 80)
    print(f"{'Time Saved':<20} {time_saved:<20.2f} {format_time(abs(time_saved)):<20}")
    print(f"{'Speedup':<20} {speedup:<20.2f}x")
    print(f"{'Faster by':<20} {percent_faster:<20.1f}%")

    print("\n" + "-" * 80)
    print("AVERAGE TIME PER EPOCH")
    print("-" * 80)
    print(f"{'Traditional':<20} {trad_avg_epoch:.2f} seconds")
    print(f"{'ALaST':<20} {alast_avg_epoch:.2f} seconds")
    print(f"{'Difference':<20} {trad_avg_epoch - alast_avg_epoch:.2f} seconds per epoch")

    print("\n" + "-" * 80)
    print("NUMBER OF EPOCHS")
    print("-" * 80)
    print(f"{'Traditional':<20} {len(trad_metrics['time_per_epoch'])} epochs")
    print(f"{'ALaST':<20} {len(alast_metrics['time_per_epoch'])} epochs")

    print("\n" + "-" * 80)
    print("ACCURACY COMPARISON")
    print("-" * 80)
    print(f"{'Traditional Best':<20} {trad_best_acc:.2f}%")
    print(f"{'ALaST Best':<20} {alast_best_acc:.2f}%")
    print(f"{'Difference':<20} {acc_diff:+.2f}%")

    print("\n" + "-" * 80)
    print("EFFICIENCY METRICS")
    print("-" * 80)

    # Memory comparison if available
    if 'peak_memory' in trad_metrics and 'peak_memory' in alast_metrics:
        if trad_metrics['peak_memory'] and alast_metrics['peak_memory']:
            trad_mem = max(trad_metrics['peak_memory'])
            alast_mem = max(alast_metrics['peak_memory'])
            mem_saved = trad_mem - alast_mem
            mem_percent = (mem_saved / trad_mem * 100) if trad_mem > 0 else 0

            print(f"{'Peak Memory (Trad)':<20} {trad_mem:.2f} GB")
            print(f"{'Peak Memory (ALaST)':<20} {alast_mem:.2f} GB")
            print(f"{'Memory Saved':<20} {mem_saved:.2f} GB ({mem_percent:.1f}%)")

    # Layer freezing information (ALaST only)
    if 'frozen_layers' in alast_metrics:
        final_frozen = len(alast_metrics['frozen_layers'][-1])
        total_layers = 12  # Assuming ViT-Base with 12 layers
        print(f"\n{'ALaST Active Layers':<20} {total_layers - final_frozen}/{total_layers}")
        print(f"{'ALaST Frozen Layers':<20} {final_frozen}/{total_layers}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if speedup > 1:
        print(f"✓ ALaST is {speedup:.2f}x FASTER than traditional fine-tuning")
        print(f"✓ Time saved: {format_time(abs(time_saved))}")
    else:
        print(f"✗ ALaST is {1 / speedup:.2f}x SLOWER than traditional fine-tuning")

    if acc_diff >= -0.5:
        print(f"✓ ALaST maintains comparable accuracy ({acc_diff:+.2f}%)")
    else:
        print(f"⚠ ALaST has lower accuracy ({acc_diff:+.2f}%)")

    # Overall assessment
    print("\nOverall Assessment:")
    if speedup > 1.1 and acc_diff >= -1.0:
        print("  🎉 ALaST provides significant efficiency gains with maintained accuracy!")
    elif speedup > 1.0 and acc_diff >= -2.0:
        print("  👍 ALaST provides modest efficiency gains with acceptable accuracy.")
    else:
        print("  ⚠ Consider adjusting ALaST hyperparameters for better efficiency.")

    print("=" * 80)

    return {
        'traditional_total': trad_total,
        'alast_total': alast_total,
        'speedup': speedup,
        'time_saved': time_saved,
        'percent_faster': percent_faster,
        'traditional_accuracy': trad_best_acc,
        'alast_accuracy': alast_best_acc,
        'accuracy_difference': acc_diff
    }


if __name__ == "__main__":
    # You can specify custom paths if needed
    import sys

    if len(sys.argv) == 3:
        trad_path = sys.argv[1]
        alast_path = sys.argv[2]
        compare_training_times(trad_path, alast_path)
    else:
        # Use default paths
        compare_training_times()