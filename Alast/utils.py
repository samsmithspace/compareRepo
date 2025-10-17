import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch


def generate_comparison_plots(trad_metrics, alast_metrics, alast_model):
    """Generate comparison plots between traditional and ALaST training"""
    os.makedirs('results', exist_ok=True)

    # 1. Validation Accuracy Comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(trad_metrics['val_acc'], 'b-', marker='o', label='Traditional')
    plt.plot(alast_metrics['val_acc'], 'r-', marker='x', label='ALaST')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 2. Training Loss Comparison
    plt.subplot(2, 2, 2)
    plt.plot(trad_metrics['train_loss'], 'b-', marker='o', label='Traditional')
    plt.plot(alast_metrics['train_loss'], 'r-', marker='x', label='ALaST')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 3. Training Time Comparison
    plt.subplot(2, 2, 3)
    trad_time = np.array(trad_metrics['time_per_epoch'])
    alast_time = np.array(alast_metrics['time_per_epoch'])
    epochs = np.arange(len(trad_time))

    plt.bar(epochs - 0.2, trad_time, width=0.4, label='Traditional', alpha=0.8)
    plt.bar(epochs + 0.2, alast_time, width=0.4, label='ALaST', alpha=0.8)
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Seconds')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 4. Memory Usage Comparison
    plt.subplot(2, 2, 4)
    trad_mem = np.array(trad_metrics['peak_memory'])
    alast_mem = np.array(alast_metrics['peak_memory'])

    plt.bar(epochs - 0.2, trad_mem, width=0.4, label='Traditional', alpha=0.8)
    plt.bar(epochs + 0.2, alast_mem, width=0.4, label='ALaST', alpha=0.8)
    plt.title('Peak Memory Usage')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (GB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig('results/traditional_vs_alast_beans_comparison.png')

    # 5. Layer Budget Evolution for ALaST
    plt.figure(figsize=(10, 6))
    layer_budgets = np.array(alast_metrics['layer_budgets'])

    for i in range(layer_budgets.shape[1]):
        plt.plot(layer_budgets[:, i], label=f'Layer {i + 1}')

    plt.title('ALaST Layer Budget Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Budget Value')
    plt.ylim(0.7, 1.05)  # Based on min budget in ALaST implementation
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/alast_beans_layer_budgets.png')

    # 6. Frozen Layers Visualization
    plt.figure(figsize=(10, 6))
    frozen_layers_history = alast_metrics['frozen_layers']

    epoch_nums = list(range(len(frozen_layers_history)))
    active_layers_count = [alast_model.num_layers - len(frozen) for frozen in frozen_layers_history]

    plt.plot(epoch_nums, active_layers_count, 'g-o')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Active Layers')
    plt.title('Active Layers Throughout Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/alast_beans_active_layers.png')

    # Print summary statistics
    print("\n===== Performance Summary =====")

    # Accuracy comparison
    trad_best_acc = max(trad_metrics['val_acc'])
    alast_best_acc = max(alast_metrics['val_acc'])
    print(f"Traditional - Best Accuracy: {trad_best_acc:.2f}%")
    print(f"ALaST - Best Accuracy: {alast_best_acc:.2f}%")
    print(f"Accuracy Difference: {alast_best_acc - trad_best_acc:.2f}%")

    # Speed comparison
    avg_trad_time = np.mean(trad_metrics['time_per_epoch'])
    avg_alast_time = np.mean(alast_metrics['time_per_epoch'])
    speedup = avg_trad_time / avg_alast_time
    print(f"Average Training Time - Traditional: {avg_trad_time:.2f}s, ALaST: {avg_alast_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    # Memory comparison
    avg_trad_mem = np.mean(trad_metrics['peak_memory'])
    avg_alast_mem = np.mean(alast_metrics['peak_memory'])
    mem_reduction = (avg_trad_mem - avg_alast_mem) / avg_trad_mem * 100
    print(f"Average Memory Usage - Traditional: {avg_trad_mem:.2f}GB, ALaST: {avg_alast_mem:.2f}GB")
    print(f"Memory Reduction: {mem_reduction:.2f}%")

    # Calculate active layers and token efficiency
    final_frozen = len(alast_metrics['frozen_layers'][-1])
    final_active = alast_model.num_layers - final_frozen
    print(
        f"Final Active Layers: {final_active}/{alast_model.num_layers} ({final_active / alast_model.num_layers * 100:.1f}%)")

    # Calculate theoretical computational savings
    avg_budget = np.mean(alast_metrics['layer_budgets'][-1])
    token_ratio = (avg_budget + alast_model.min_token_percent) / 2  # Approximation of average token ratio
    layer_ratio = final_active / alast_model.num_layers
    theoretical_saving = (1 - token_ratio * layer_ratio) * 100
    print(f"Estimated Computational Savings: {theoretical_saving:.1f}%")

    print("\nComparison plots saved to results/ directory")

    # Save detailed comparison results as JSON
    try:
        comparison_results = {
            'accuracy': {
                'traditional_final': trad_metrics['val_acc'][-1],
                'alast_final': alast_metrics['val_acc'][-1],
                'traditional_best': trad_best_acc,
                'alast_best': alast_best_acc,
                'difference': alast_best_acc - trad_best_acc,
                'percent_diff': (alast_best_acc - trad_best_acc) / trad_best_acc * 100
            },
            'training_time': {
                'traditional_avg': float(avg_trad_time),
                'alast_avg': float(avg_alast_time),
                'speedup': float(speedup),
                'percent_reduction': float((avg_trad_time - avg_alast_time) / avg_trad_time * 100)
            },
            'memory': {
                'traditional_avg': float(avg_trad_mem),
                'alast_avg': float(avg_alast_mem),
                'reduction': float(mem_reduction)
            },
            'layers': {
                'total_layers': alast_model.num_layers,
                'final_active_layers': final_active,
                'warmup_epochs': alast_model.warmup_epochs,
                'n_train_layers': alast_model.n_train_layers
            },
            'efficiency': {
                'estimated_computational_savings': float(theoretical_saving),
                'final_layer_budgets': alast_model.beta.cpu().numpy().tolist(),
                'min_token_percent': alast_model.min_token_percent
            }
        }

        # Save comparison results
        with open('results/beans_comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)

        print("Detailed comparison metrics saved to results/beans_comparison_results.json")
    except Exception as e:
        print(f"Error saving comparison results: {str(e)}")


def plot_training_metrics(metrics, title="Beans"):
    """Plot training metrics for a single model"""
    try:
        plt.figure(figsize=(15, 10))

        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(metrics['val_acc'], 'o-')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(metrics['train_loss'], 'o-')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot time per epoch
        plt.subplot(2, 2, 3)
        plt.bar(range(len(metrics['time_per_epoch'])), metrics['time_per_epoch'])
        plt.title('Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Plot layer budgets if available
        if 'layer_budgets' in metrics:
            plt.subplot(2, 2, 4)
            layer_budgets = np.array(metrics['layer_budgets'])
            for i in range(layer_budgets.shape[1]):
                plt.plot(layer_budgets[:, i], label=f'Layer {i + 1}')
            plt.title('Layer Budgets Evolution')
            plt.xlabel('Epoch')
            plt.ylabel('Budget Value')
            plt.ylim(0.7, 1.05)  # Based on min budget
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(f'results/alast_{title.lower()}_training_metrics.png')
        print(f"Training metrics plot saved to results/alast_{title.lower()}_training_metrics.png")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")