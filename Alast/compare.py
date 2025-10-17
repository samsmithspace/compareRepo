import torch
import torch.nn as nn
from torch.optim import Adam
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from tqdm import tqdm
import json
import os


class MetricsTracker:
    """Helper class to track training metrics"""

    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_acc': [],
            'time_per_epoch': [],
            'peak_memory': [],
            'throughput': [],  # samples/second
            'flops_per_epoch': []
        }

    def update(self, metric_name, value):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_metrics(self):
        return self.metrics

    def save(self, filename):
        # Convert to regular Python types for JSON serialization
        serializable_metrics = {}
        for k, v in self.metrics.items():
            serializable_metrics[k] = [float(x) if isinstance(x, torch.Tensor) else x for x in v]

        with open(filename, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)


def train_traditional(model, train_loader, val_loader, num_epochs=10, lr=1e-4,
                      device='cuda', mixed_precision=True, save_path=None):
    """
    Standard fine-tuning for Vision Transformer models

    Args:
        model: Vision Transformer model to fine-tune
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        mixed_precision: Whether to use mixed precision training
        save_path: Where to save the trained model

    Returns:
        trained_model: Trained model
        metrics: Dictionary of training metrics
    """
    model = model.to(device)

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Setup for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

    # Initialize metrics tracker
    tracker = MetricsTracker()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        # Reset CUDA memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Track number of samples processed
        num_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            num_samples += batch_size

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)

                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        # Record metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        throughput = num_samples / epoch_time  # samples/second

        tracker.update('train_loss', avg_loss)
        tracker.update('time_per_epoch', epoch_time)
        tracker.update('throughput', throughput)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            tracker.update('peak_memory', peak_mem)
        else:
            peak_mem = 0

        # Validation
        val_acc = evaluate(model, val_loader, device, mixed_precision)
        tracker.update('val_acc', val_acc)

        # Estimate computational cost (FLOPs)
        # This is a simplified approximation - use a proper FLOP counter for exact measurements
        estimate_flops(model, tracker, train_loader)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%, "
              f"Time={epoch_time:.2f}s, Speed={throughput:.1f} samples/s, "
              f"Memory={peak_mem:.2f}GB")

    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        # Save metrics next to model
        metrics_path = save_path.replace('.pth', '_metrics.json')
        tracker.save(metrics_path)

    return model, tracker.get_metrics()


def evaluate(model, dataloader, device, mixed_precision=True):
    """Evaluate model accuracy on dataloader"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


def estimate_flops(model, tracker, dataloader):
    """Estimate FLOPs for a ViT model (simplified approximation)"""
    # Extract model parameters
    if hasattr(model, 'embed_dim'):
        hidden_dim = model.embed_dim
    elif hasattr(model, 'hidden_size'):
        hidden_dim = model.hidden_size
    else:
        hidden_dim = 768  # Default for ViT-B

    if hasattr(model, 'blocks') and isinstance(model.blocks, (list, nn.ModuleList)):
        num_layers = len(model.blocks)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        num_layers = len(model.encoder.layer)
    else:
        num_layers = 12  # Default for ViT-B

    # Estimate sequence length based on patch size
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'patch_size'):
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple):
            pass
        else:
            patch_size = (16, 16)  # Default
    else:
        patch_size = (16, 16)  # Default

    # Typical ViT input size
    image_size = 224

    # Calculate number of patches
    if isinstance(patch_size, tuple):
        num_patches = (image_size // patch_size[0]) * (image_size // patch_size[1])
    else:
        num_patches = (image_size // patch_size) ** 2

    # Sequence length includes class token
    seq_len = num_patches + 1

    # Approximate FLOPs for self-attention and MLP in each layer
    flops_per_sample = num_layers * (
        # Self-attention: 4 * hidden_dim * seq_len^2 (QKV projections and attention matrix)
            4 * hidden_dim * seq_len ** 2 +
            # MLP: 8 * hidden_dim^2 * seq_len (two linear layers with 4x expansion)
            8 * hidden_dim ** 2 * seq_len
    )

    # Total FLOPs for the dataset
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    total_flops = flops_per_sample * batch_size * num_batches

    # Update metrics
    tracker.update('flops_per_epoch', total_flops)

    return total_flops


def main():
    """Run traditional fine-tuning and save metrics for comparison"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Food-101 dataset
    print("Loading Food-101 dataset...")
    train_dataset = datasets.Food101(
        root='./data', split='train', download=True, transform=transform_train
    )
    val_dataset = datasets.Food101(
        root='./data', split='test', download=True, transform=transform_val
    )

    dataset_name = "food101"
    num_classes = 101

    # Optionally use a subset for faster experimentation
    # indices = torch.randperm(len(train_dataset))[:20000]
    # train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Prepare dataloaders
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataset: {dataset_name} with {len(train_dataset)} training and {len(val_dataset)} validation images")

    # Create model
    print("Creating model...")
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    # Training parameters
    num_epochs = 10
    learning_rate = 1e-4
    mixed_precision = torch.cuda.is_available()  # Use mixed precision if CUDA available

    print(f"Starting traditional fine-tuning for {num_epochs} epochs with lr={learning_rate}")

    # Train with traditional fine-tuning
    trained_model, metrics = train_traditional(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        mixed_precision=mixed_precision,
        save_path=f'results/traditional_{dataset_name}.pth'
    )

    # Save metrics for comparison
    metrics_file = f'results/traditional_{dataset_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert all values to standard Python types
        serializable_metrics = {}
        for k, v in metrics.items():
            serializable_metrics[k] = [float(x) if isinstance(x, torch.Tensor) else x for x in v]
        json.dump(serializable_metrics, f, indent=2)

    print(f"Training complete! Metrics saved to {metrics_file}")
    print(f"Final validation accuracy: {metrics['val_acc'][-1]:.2f}%")
    print(f"Average epoch time: {np.mean(metrics['time_per_epoch']):.2f}s")
    if 'peak_memory' in metrics and metrics['peak_memory']:
        print(f"Average peak memory: {np.mean(metrics['peak_memory']):.2f}GB")

    print("\nTo compare with ALaST, run your ALaST implementation and then use:")
    print("python compare_methods.py")


if __name__ == "__main__":
    main()
