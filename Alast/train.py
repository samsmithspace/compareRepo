import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import numpy as np
import os
import json
import traceback
from torch import nn

from models import ImprovedALaST
from evaluate import evaluate


def train_alast(model, train_loader, val_loader, num_epochs=10, lr=1e-4,
                device='cuda', n_train_layers=9, mixed_precision=True):
    """
    Train a ViT model using improved ALaST for efficient fine-tuning
    with enhanced stability after warmup
    """
    # Initialize ALaST wrapper with improved importance estimation
    alast_model = ImprovedALaST(model, n_train_layers=n_train_layers, alpha=0.005)
    alast_model = alast_model.to(device)

    # Enable diagnostics for the first couple of epochs
    alast_model.print_diagnostics = True

    # Optimizer and loss
    optimizer = Adam(filter(lambda p: p.requires_grad, alast_model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision setup
    if mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    # Metrics tracking
    metrics = {
        'train_loss': [],
        'val_acc': [],
        'time_per_epoch': [],
        'peak_memory': [],
        'layer_budgets': [],
        'frozen_layers': []
    }

    # Training loop
    best_accuracy = 0
    best_model_state = None
    prev_val_acc = 0

    for epoch in range(num_epochs):
        alast_model.current_epoch = epoch  # Update epoch counter in model
        alast_model.batch_idx = 0  # Reset batch counter

        # Turn off diagnostics after a few epochs
        if epoch >= 7:
            alast_model.print_diagnostics = False

        alast_model.train()
        total_loss = 0
        start_time = time.time()

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if mixed_precision and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits, deltas = alast_model(images, compute_deltas=True)
                    loss = criterion(logits, labels)

                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, deltas = alast_model(images, compute_deltas=True)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            # Budget updates and layer freezing
            alast_model.update_budgets(deltas)
            alast_model.update_frozen_layers()

            total_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Record metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_loss)
        metrics['time_per_epoch'].append(epoch_time)
        metrics['layer_budgets'].append(alast_model.beta.cpu().numpy().copy())
        metrics['frozen_layers'].append(list(alast_model.frozen_layers))

        # Memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            metrics['peak_memory'].append(peak_mem)
        else:
            peak_mem = 0

        # Validation
        val_acc = evaluate(alast_model, val_loader, device, mixed_precision)
        metrics['val_acc'].append(val_acc)

        # Check for accuracy drops after warmup
        if epoch > alast_model.warmup_epochs and val_acc < prev_val_acc * 0.95:
            print(f"Warning: >5% accuracy drop detected! ({prev_val_acc:.2f}% → {val_acc:.2f}%)")
            print("Consider unfreezing some layers or adjusting token dropping parameters.")

            # Emergency recovery mechanism for dramatic accuracy drops
            if val_acc < prev_val_acc * 0.9:  # >10% drop
                print("Significant accuracy drop! Implementing emergency recovery...")

                # 1. Restore model from previous best checkpoint if available
                if best_model_state is not None:
                    print("Restoring from previous best checkpoint")
                    alast_model.load_state_dict(best_model_state)

                # 2. Reduce alpha to slow down budget changes
                alast_model.alpha *= 0.5
                print(f"Reduced budget learning rate to {alast_model.alpha:.6f}")

                # 3. Unfreeze more layers
                current_frozen = len(alast_model.frozen_layers)
                if current_frozen > 0:
                    # Select a frozen layer to unfreeze (preferably early layer)
                    layers_to_unfreeze = min(2, current_frozen)  # Unfreeze up to 2 layers

                    # Sort frozen layers by importance (descending)
                    frozen_importance = [(i, alast_model.layer_importance[i].item())
                                         for i in alast_model.frozen_layers]
                    frozen_importance.sort(key=lambda x: x[1], reverse=True)

                    # Unfreeze the most important frozen layers
                    for i in range(layers_to_unfreeze):
                        if i < len(frozen_importance):
                            layer_idx = frozen_importance[i][0]
                            print(f"Unfreezing layer {layer_idx}")
                            for param in alast_model.blocks[layer_idx].parameters():
                                param.requires_grad_(True)
                            alast_model.frozen_layers.remove(layer_idx)

                            # Also increase its budget
                            alast_model.beta[layer_idx] = torch.max(alast_model.beta)

                # 4. Increase min token percentage
                alast_model.min_token_percent = min(0.95, alast_model.min_token_percent + 0.05)
                print(f"Increased minimum token percentage to {alast_model.min_token_percent:.2f}")

                # 5. Adjust optimizer
                optimizer = Adam(filter(lambda p: p.requires_grad, alast_model.parameters()), lr=lr * 0.8)
                if scaler is not None:
                    scaler = torch.amp.GradScaler('cuda')  # Reset scaler

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save a copy of the model state
            best_model_state = {k: v.cpu().clone() for k, v in alast_model.state_dict().items()}

        # Update previous accuracy for next epoch's comparison
        prev_val_acc = val_acc

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%, "
              f"Time={epoch_time:.2f}s, Memory={peak_mem:.2f}GB")

        # Print budget information
        print(f"Layer budgets: {alast_model.beta.cpu().numpy().round(3)}")
        print(f"Active layers: {sorted(set(range(len(alast_model.blocks))) - alast_model.frozen_layers)}")

    # Restore best model
    if best_model_state is not None:
        alast_model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_accuracy:.2f}%")

    # Save final metrics safely
    os.makedirs('results', exist_ok=True)

    try:
        with open('results/alast_beans_metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for k, v in metrics.items():
                if k == 'layer_budgets':
                    serializable_metrics[k] = [b.tolist() if hasattr(b, 'tolist') else b for b in v]
                else:
                    serializable_metrics[k] = [float(x) if isinstance(x, (np.number, torch.Tensor)) else x for x in v]
            json.dump(serializable_metrics, f, indent=2)
        print("ALaST metrics saved successfully to results/alast_beans_metrics.json")
    except Exception as e:
        print(f"Error saving ALaST metrics: {str(e)}")
        traceback.print_exc()

    return alast_model, metrics, best_accuracy


def train_traditional_vit(model, train_loader, val_loader, num_epochs=10, lr=1e-4,
                          device='cuda', mixed_precision=True):
    """
    Standard fine-tuning for Vision Transformer models (for comparison)
    """
    model = model.to(device)

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision setup
    if mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    # Metrics tracking
    metrics = {
        'train_loss': [],
        'val_acc': [],
        'time_per_epoch': [],
        'peak_memory': []
    }

    # Training loop
    best_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        # Reset CUDA memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if mixed_precision and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
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

        # Step the scheduler
        scheduler.step()

        # Record metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_loss)
        metrics['time_per_epoch'].append(epoch_time)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            metrics['peak_memory'].append(peak_mem)
        else:
            peak_mem = 0

        # Validation
        val_acc = evaluate(model, val_loader, device, mixed_precision)
        metrics['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%, "
              f"Time={epoch_time:.2f}s, Memory={peak_mem:.2f}GB")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_accuracy:.2f}%")

    # Save metrics safely
    os.makedirs('results', exist_ok=True)
    try:
        with open('results/traditional_beans_metrics.json', 'w') as f:
            serializable_metrics = {}
            for k, v in metrics.items():
                serializable_metrics[k] = [float(x) if isinstance(x, (np.number, torch.Tensor)) else x for x in v]
            json.dump(serializable_metrics, f, indent=2)
    except Exception as e:
        print(f"Error saving traditional metrics: {str(e)}")
        traceback.print_exc()

    return model, metrics, best_accuracy