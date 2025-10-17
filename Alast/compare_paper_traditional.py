"""
Comparison script for Paper ALaST implementation vs Traditional ViT fine-tuning
Runs both methods on CIFAR-100 and generates comprehensive comparison metrics
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from timm import create_model
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

# Import the paper's ALaST implementation
from paperalast import ALaSTViT, evaluate as paper_evaluate


def train_traditional_vit(num_epochs=20, batch_size=128, lr=0.0001, device='cuda'):
    """
    Standard fine-tuning for Vision Transformer on CIFAR-100

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on

    Returns:
        model: Trained model
        metrics: Dictionary of training metrics
    """
    print("=" * 80)
    print("TRADITIONAL VIT FINE-TUNING")
    print("=" * 80)

    # Data transformations (matching paper setup)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-100
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Create model - using timm's ViT-Base to match dimensions
    # Note: Paper uses DeiT-S (384 dim, 12 layers, 6 heads)
    # For fair comparison, we use the same architecture
    print("\nCreating Vision Transformer model...")
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Metrics tracking
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'time_per_epoch': [],
        'peak_memory': [],
        'total_training_time': 0
    }

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {lr}, Batch size: {batch_size}")
    print("-" * 80)

    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            metrics['peak_memory'].append(peak_mem)
        else:
            peak_mem = 0

        # Evaluate on test set
        test_acc = evaluate_traditional(model, test_loader, device)

        # Record metrics
        metrics['train_loss'].append(avg_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['time_per_epoch'].append(epoch_time)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | Memory: {peak_mem:.2f}GB")
        print("-" * 80)

    # Total training time
    metrics['total_training_time'] = time.time() - total_start_time

    print(f"\nTraining complete!")
    print(f"Total time: {metrics['total_training_time']:.2f}s")
    print(f"Final test accuracy: {metrics['test_acc'][-1]:.2f}%")
    print(f"Best test accuracy: {max(metrics['test_acc']):.2f}%")

    return model, metrics, test_loader


def evaluate_traditional(model, loader, device):
    """Evaluate traditional model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def train_paper_alast(num_epochs=20, batch_size=128, lr=0.0001, K=9, device='cuda'):
    """
    Train using Paper's ALaST implementation

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate (also used as alpha for budget updates)
        K: Number of trainable layers per iteration
        device: Device to train on

    Returns:
        model: Trained ALaST model
        metrics: Dictionary of training metrics
    """
    print("=" * 80)
    print("PAPER ALAST IMPLEMENTATION")
    print("=" * 80)

    # Data transformations (same as traditional)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-100
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Create ALaST model (DeiT-S architecture from paper)
    print("\nCreating ALaST Vision Transformer model...")
    model = ALaSTViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=100,
        embed_dim=384,  # DeiT-S
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        K=K,
        alpha=lr,  # Paper: α = fine-tuning learning rate
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and loss
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Metrics tracking
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'time_per_epoch': [],
        'peak_memory': [],
        'budgets_history': [],
        'frozen_layers_history': [],
        'active_layers_history': [],
        'total_training_time': 0
    }

    print(f"\nStarting ALaST training for {num_epochs} epochs...")
    print(f"Learning rate: {lr}, Batch size: {batch_size}, K: {K}")
    print("-" * 80)

    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)

            # Forward pass with delta computation
            logits, deltas = model(images, compute_deltas=True)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update budgets and select layers (per-batch, as in paper)
            model.update_budgets_and_select_layers(deltas)

            # Rebuild optimizer with new trainable parameters
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

            # Statistics
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            metrics['peak_memory'].append(peak_mem)
        else:
            peak_mem = 0

        # Evaluate on test set
        test_acc = paper_evaluate(model, test_loader, device)

        # Get budget info
        budget_info = model.get_budget_info()

        # Record metrics
        metrics['train_loss'].append(avg_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['time_per_epoch'].append(epoch_time)
        metrics['budgets_history'].append(budget_info['budgets'].tolist())
        metrics['frozen_layers_history'].append(budget_info['frozen_layers'])
        metrics['active_layers_history'].append(len(budget_info['trainable_layers']))

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | Memory: {peak_mem:.2f}GB")
        print(f"  Active layers: {len(budget_info['trainable_layers'])}/{model.depth}")
        print(f"  Frozen layers: {budget_info['frozen_layers']}")
        print("-" * 80)

    # Total training time
    metrics['total_training_time'] = time.time() - total_start_time

    print(f"\nALaST training complete!")
    print(f"Total time: {metrics['total_training_time']:.2f}s")
    print(f"Final test accuracy: {metrics['test_acc'][-1]:.2f}%")
    print(f"Best test accuracy: {max(metrics['test_acc']):.2f}%")

    return model, metrics, test_loader


def generate_comparison_plots(trad_metrics, alast_metrics, output_dir='results'):
    """Generate comprehensive comparison plots"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Main comparison plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Test Accuracy
    axes[0, 0].plot(trad_metrics['test_acc'], 'b-o', label='Traditional', linewidth=2, markersize=6)
    axes[0, 0].plot(alast_metrics['test_acc'], 'r-x', label='Paper ALaST', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Training Loss
    axes[0, 1].plot(trad_metrics['train_loss'], 'b-o', label='Traditional', linewidth=2, markersize=6)
    axes[0, 1].plot(alast_metrics['train_loss'], 'r-x', label='Paper ALaST', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Training Loss', fontsize=12)
    axes[0, 1].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Time per Epoch
    epochs = np.arange(len(trad_metrics['time_per_epoch']))
    width = 0.35
    axes[1, 0].bar(epochs - width / 2, trad_metrics['time_per_epoch'], width,
                   label='Traditional', alpha=0.8, color='blue')
    axes[1, 0].bar(epochs + width / 2, alast_metrics['time_per_epoch'], width,
                   label='Paper ALaST', alpha=0.8, color='red')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Memory Usage
    axes[1, 1].bar(epochs - width / 2, trad_metrics['peak_memory'], width,
                   label='Traditional', alpha=0.8, color='blue')
    axes[1, 1].bar(epochs + width / 2, alast_metrics['peak_memory'], width,
                   label='Paper ALaST', alpha=0.8, color='red')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Memory (GB)', fontsize=12)
    axes[1, 1].set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_alast_vs_traditional_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/paper_alast_vs_traditional_comparison.png")

    # 2. ALaST-specific plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Budget Evolution
    budgets_array = np.array(alast_metrics['budgets_history'])
    for i in range(budgets_array.shape[1]):
        axes[0].plot(budgets_array[:, i], label=f'Layer {i + 1}', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Budget Value', fontsize=12)
    axes[0].set_title('ALaST Layer Budget Evolution', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Active Layers Over Time
    axes[1].plot(alast_metrics['active_layers_history'], 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Number of Active Layers', fontsize=12)
    axes[1].set_title('Active Layers Throughout Training', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=9, color='r', linestyle='--', label='Target K=9', linewidth=2)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_alast_layer_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/paper_alast_layer_analysis.png")

    plt.close('all')


def print_comparison_summary(trad_metrics, alast_metrics):
    """Print detailed comparison summary"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 80)

    # Accuracy metrics
    print("\n" + "-" * 80)
    print("ACCURACY METRICS")
    print("-" * 80)
    trad_best = max(trad_metrics['test_acc'])
    trad_final = trad_metrics['test_acc'][-1]
    alast_best = max(alast_metrics['test_acc'])
    alast_final = alast_metrics['test_acc'][-1]

    print(f"{'Method':<20} {'Best Acc':<15} {'Final Acc':<15}")
    print(f"{'Traditional':<20} {trad_best:<15.2f}% {trad_final:<15.2f}%")
    print(f"{'Paper ALaST':<20} {alast_best:<15.2f}% {alast_final:<15.2f}%")
    print(f"{'Difference':<20} {alast_best - trad_best:<15.2f}% {alast_final - trad_final:<15.2f}%")

    # Training time metrics
    print("\n" + "-" * 80)
    print("TRAINING TIME METRICS")
    print("-" * 80)
    trad_total = trad_metrics['total_training_time']
    alast_total = alast_metrics['total_training_time']
    trad_avg = np.mean(trad_metrics['time_per_epoch'])
    alast_avg = np.mean(alast_metrics['time_per_epoch'])

    speedup = trad_total / alast_total
    time_saved = trad_total - alast_total

    print(f"{'Method':<20} {'Total Time':<20} {'Avg per Epoch':<20}")
    print(f"{'Traditional':<20} {trad_total:<20.2f}s {trad_avg:<20.2f}s")
    print(f"{'Paper ALaST':<20} {alast_total:<20.2f}s {alast_avg:<20.2f}s")
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time Saved: {time_saved:.2f}s ({time_saved / 60:.2f} minutes)")
    print(f"Percentage Faster: {((trad_total - alast_total) / trad_total * 100):.2f}%")

    # Memory metrics
    print("\n" + "-" * 80)
    print("MEMORY METRICS")
    print("-" * 80)
    trad_mem_avg = np.mean(trad_metrics['peak_memory'])
    alast_mem_avg = np.mean(alast_metrics['peak_memory'])
    trad_mem_max = max(trad_metrics['peak_memory'])
    alast_mem_max = max(alast_metrics['peak_memory'])

    mem_saved = trad_mem_avg - alast_mem_avg
    mem_reduction = (mem_saved / trad_mem_avg * 100) if trad_mem_avg > 0 else 0

    print(f"{'Method':<20} {'Avg Memory':<20} {'Peak Memory':<20}")
    print(f"{'Traditional':<20} {trad_mem_avg:<20.2f}GB {trad_mem_max:<20.2f}GB")
    print(f"{'Paper ALaST':<20} {alast_mem_avg:<20.2f}GB {alast_mem_max:<20.2f}GB")
    print(f"\nMemory Saved: {mem_saved:.2f}GB")
    print(f"Memory Reduction: {mem_reduction:.2f}%")

    # ALaST-specific metrics
    if 'active_layers_history' in alast_metrics:
        print("\n" + "-" * 80)
        print("ALAST-SPECIFIC METRICS")
        print("-" * 80)
        final_active = alast_metrics['active_layers_history'][-1]
        final_frozen = 12 - final_active  # Assuming 12 layers
        avg_active = np.mean(alast_metrics['active_layers_history'])

        print(f"Final Active Layers: {final_active}/12 ({final_active / 12 * 100:.1f}%)")
        print(f"Final Frozen Layers: {final_frozen}/12 ({final_frozen / 12 * 100:.1f}%)")
        print(f"Average Active Layers: {avg_active:.2f}")

        # Estimate computational savings
        final_budget = np.mean(alast_metrics['budgets_history'][-1])
        layer_ratio = final_active / 12
        token_ratio = final_budget
        est_computation = layer_ratio * token_ratio
        est_savings = (1 - est_computation) * 100

        print(f"\nFinal Average Budget: {final_budget:.3f}")
        print(f"Estimated Computational Savings: {est_savings:.1f}%")

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    if speedup > 1.0 and alast_best >= trad_best - 1.0:
        print("✓ Paper ALaST provides efficiency gains while maintaining comparable accuracy!")
        print(f"  - {speedup:.2f}x faster training")
        print(f"  - {mem_reduction:.1f}% memory reduction")
        print(f"  - Accuracy difference: {alast_best - trad_best:+.2f}%")
    elif speedup > 1.0:
        print("⚠ Paper ALaST is faster but with some accuracy trade-off")
        print(f"  - {speedup:.2f}x faster training")
        print(f"  - Accuracy gap: {alast_best - trad_best:.2f}%")
    else:
        print("⚠ Paper ALaST needs hyperparameter tuning for better efficiency")

    print("=" * 80)


def save_metrics_to_json(trad_metrics, alast_metrics, output_dir='results'):
    """Save all metrics to JSON files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save traditional metrics
    with open(os.path.join(output_dir, 'traditional_cifar100_metrics.json'), 'w') as f:
        json.dump(trad_metrics, f, indent=2)
    print(f"\nSaved: {output_dir}/traditional_cifar100_metrics.json")

    # Save ALaST metrics
    with open(os.path.join(output_dir, 'paper_alast_cifar100_metrics.json'), 'w') as f:
        json.dump(alast_metrics, f, indent=2)
    print(f"Saved: {output_dir}/paper_alast_cifar100_metrics.json")

    # Save comparison summary
    comparison = {
        'accuracy': {
            'traditional_best': max(trad_metrics['test_acc']),
            'traditional_final': trad_metrics['test_acc'][-1],
            'alast_best': max(alast_metrics['test_acc']),
            'alast_final': alast_metrics['test_acc'][-1],
            'difference_best': max(alast_metrics['test_acc']) - max(trad_metrics['test_acc']),
            'difference_final': alast_metrics['test_acc'][-1] - trad_metrics['test_acc'][-1]
        },
        'training_time': {
            'traditional_total': trad_metrics['total_training_time'],
            'alast_total': alast_metrics['total_training_time'],
            'speedup': trad_metrics['total_training_time'] / alast_metrics['total_training_time'],
            'time_saved_seconds': trad_metrics['total_training_time'] - alast_metrics['total_training_time']
        },
        'memory': {
            'traditional_avg': float(np.mean(trad_metrics['peak_memory'])),
            'alast_avg': float(np.mean(alast_metrics['peak_memory'])),
            'reduction_gb': float(np.mean(trad_metrics['peak_memory']) - np.mean(alast_metrics['peak_memory'])),
            'reduction_percent': float(
                (np.mean(trad_metrics['peak_memory']) - np.mean(alast_metrics['peak_memory'])) / np.mean(
                    trad_metrics['peak_memory']) * 100)
        }
    }

    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {output_dir}/comparison_summary.json")


def main():
    """Main comparison function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    # Common hyperparameters (from paper)
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.0001
    K = 9  # Number of trainable layers for ALaST

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # 1. Train Traditional ViT
    print("\n" + "=" * 80)
    print("STEP 1: Training Traditional ViT")
    print("=" * 80)
    trad_model, trad_metrics, test_loader = train_traditional_vit(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        device=device
    )

    # Save traditional model
    torch.save(trad_model.state_dict(), 'results/traditional_cifar100_vit.pth')
    print("\nSaved traditional model to results/traditional_cifar100_vit.pth")

    # 2. Train Paper's ALaST
    print("\n" + "=" * 80)
    print("STEP 2: Training Paper's ALaST Implementation")
    print("=" * 80)
    alast_model, alast_metrics, _ = train_paper_alast(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        K=K,
        device=device
    )

    # Save ALaST model
    torch.save(alast_model.state_dict(), 'results/paper_alast_cifar100_vit.pth')
    print("\nSaved ALaST model to results/paper_alast_cifar100_vit.pth")

    # 3. Generate comparison plots
    print("\n" + "=" * 80)
    print("STEP 3: Generating Comparison Plots")
    print("=" * 80)
    generate_comparison_plots(trad_metrics, alast_metrics)

    # 4. Print detailed comparison summary
    print_comparison_summary(trad_metrics, alast_metrics)

    # 5. Save all metrics to JSON
    save_metrics_to_json(trad_metrics, alast_metrics)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/traditional_cifar100_vit.pth")
    print("  - results/paper_alast_cifar100_vit.pth")
    print("  - results/traditional_cifar100_metrics.json")
    print("  - results/paper_alast_cifar100_metrics.json")
    print("  - results/comparison_summary.json")
    print("  - results/paper_alast_vs_traditional_comparison.png")
    print("  - results/paper_alast_layer_analysis.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()