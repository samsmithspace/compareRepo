import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt


class ImprovedALaST(nn.Module):
    """
    ALaST implementation with improved layer importance estimation to prevent
    accuracy drops after warmup phase.
    """

    def __init__(self, vit_model, n_train_layers=9, alpha=0.005):
        super().__init__()
        self.model = vit_model
        self.blocks = self.model.blocks
        self.num_layers = len(self.blocks)
        self.n_train_layers = min(n_train_layers, self.num_layers)

        # Initialize with all layers active
        self.register_buffer('beta', torch.ones(self.num_layers))
        self.alpha = alpha  # Budget learning rate

        # Access model components
        self.cls_token = self.model.cls_token
        self.patch_embed = self.model.patch_embed
        self.pos_embed = self.model.pos_embed
        self.norm = self.model.norm
        self.head = self.model.head

        # Tracking for stable updates - use longer history for better stability
        self.register_buffer('delta_history', torch.zeros(self.num_layers, 25))
        self.history_ptr = 0
        self.history_filled = False

        # Start with all layers active, gradually adapt
        self.frozen_layers = set()
        self.warmup_epochs = 5  # Extended warmup
        self.current_epoch = 0

        # Higher minimum token percentages to preserve information
        self.min_token_percent = 0.8  # Higher minimum (was 0.7)

        # For diagnostics
        self.batch_idx = 0
        self.print_diagnostics = False

        # For tracking importance estimations
        self.layer_importance = torch.ones(self.num_layers, device=next(self.parameters()).device)
        self.register_buffer('cumulative_deltas', torch.zeros(self.num_layers))

    def forward(self, x, compute_deltas=True):
        """Forward pass with token selection based on attention"""
        B = x.shape[0]  # Batch size
        self.batch_idx += 1

        # Initial token embedding
        x = self.patch_embed(x)

        # Add positional embeddings
        cls_token = self.cls_token.expand(B, -1, -1)
        x = x + self.pos_embed[:, 1:, :]
        cls_pos = self.pos_embed[:, 0, :].unsqueeze(1)
        cls_token = cls_token + cls_pos
        x = torch.cat([cls_token, x], dim=1)

        # Track class token changes if needed
        deltas = torch.zeros(self.num_layers, device=x.device) if compute_deltas else None
        c_prev = x[:, 0].clone()

        # Print diagnostics for the first batch of each epoch
        if self.batch_idx <= 2 and self.print_diagnostics:
            print(f"Input CLS token norm: {torch.norm(c_prev, dim=1).mean().item():.4f}")

        # Process through transformer blocks with token selection
        for i, block in enumerate(self.blocks):
            # Decide whether to run with or without gradients
            context = torch.no_grad if i in self.frozen_layers else torch.enable_grad

            with context():
                # Forward pass through the block
                x = block(x)

                # Compute class token change if needed
                if compute_deltas:
                    c_curr = x[:, 0]
                    delta = torch.norm(c_curr - c_prev, p=2)
                    deltas[i] = delta.item()
                    c_prev = c_curr.clone()

                    # Track cumulative deltas for better importance estimation
                    if self.training:
                        self.cumulative_deltas[i] += delta.item()

                    # Print layer diagnostics
                    if i == 0 and self.batch_idx <= 2 and self.print_diagnostics:
                        print(f"Layer {i} - Delta: {delta.item():.4f}, Budget: {self.beta[i].item():.2f}")

                # Only apply token dropping after warmup and not in the last layer
                if self.current_epoch >= self.warmup_epochs and i < self.num_layers - 1:
                    # Get budget for this layer (how many tokens to keep)
                    budget = self.beta[i].item()

                    # Only drop tokens if budget is meaningfully below 1.0
                    if budget < 0.98:  # More conservative threshold (was 0.95)
                        # Compute attention scores (class token to patches)
                        cls_token = x[:, 0:1]
                        patches = x[:, 1:]

                        # Simple attention approximation (dot product)
                        cls_token_norm = F.normalize(cls_token, dim=2)
                        patches_norm = F.normalize(patches, dim=2)
                        attn_scores = torch.bmm(patches_norm,
                                                cls_token_norm.transpose(1, 2)).squeeze(-1)

                        # Conservative token keeping - always keep at least min_token_percent
                        keep_percent = max(budget, self.min_token_percent)
                        k = max(int(keep_percent * (x.size(1) - 1)), 1)

                        # Get indices of top-k tokens
                        topk_indices = torch.topk(attn_scores, k=k, dim=1).indices + 1

                        # Always keep class token (index 0)
                        keep_indices = torch.zeros(B, k + 1, dtype=torch.long, device=x.device)
                        keep_indices[:, 0] = 0  # Class token
                        keep_indices[:, 1:] = topk_indices

                        # Gather selected tokens
                        x = torch.gather(x, 1,
                                         keep_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))

                        # Print token dropping diagnostics
                        if i == 0 and self.batch_idx <= 2 and self.print_diagnostics:
                            print(f"Layer {i} - Kept {k}/{x.size(1) - 1} tokens ({keep_percent:.2f})")

        # Classification head
        x = self.norm(x[:, 0])
        logits = self.head(x)

        if compute_deltas:
            return logits, deltas
        return logits

    def update_budgets(self, deltas):
        """Improved budget estimation with log scaling and min-max normalization"""
        # Don't update during warmup epochs
        if self.current_epoch < self.warmup_epochs:
            return

        # Update history buffer for stability
        with torch.no_grad():
            self.delta_history[:, self.history_ptr] = deltas
            self.history_ptr = (self.history_ptr + 1) % self.delta_history.shape[1]
            if self.history_ptr == 0:
                self.history_filled = True

            # Use history for more stable estimates
            if self.history_filled:
                # Use median of recent deltas for stability + robustness to outliers
                stable_deltas = torch.median(self.delta_history, dim=1).values

                # Apply log scaling to better differentiate important layers
                # This helps prevent small numerical differences from causing large budget changes
                log_deltas = torch.log1p(stable_deltas)

                # Normalize
                if log_deltas.sum() > 0:
                    normalized_deltas = log_deltas / (log_deltas.sum() + 1e-8)

                    # Compute relative importance with min-max normalization
                    min_delta = normalized_deltas.min()
                    delta_range = normalized_deltas.max() - min_delta

                    if delta_range > 0:
                        # Convert to [0,1] range
                        normalized_importance = (normalized_deltas - min_delta) / delta_range

                        # Now compute budget updates
                        avg_importance = 1.0 / self.num_layers
                        scaled_importance = normalized_importance - avg_importance

                        # More gradual updates right after warmup
                        epochs_after_warmup = self.current_epoch - self.warmup_epochs
                        scale_factor = min(1.0, 0.2 + 0.1 * epochs_after_warmup)

                        budget_updates = scaled_importance * (self.alpha * scale_factor)

                        # Keep high min budget early, gradually decrease
                        min_budget = max(0.85 - 0.01 * epochs_after_warmup, 0.75)

                        # Update budgets
                        self.beta.add_(budget_updates).clamp_(min_budget, 1.0)

                        # Update layer importance estimate
                        self.layer_importance = normalized_importance

    def update_frozen_layers(self):
        """Improved layer freezing that accounts for layer position"""
        # Don't freeze any layers during warmup
        if self.current_epoch < self.warmup_epochs:
            for i in range(self.num_layers):
                if i in self.frozen_layers:
                    self.frozen_layers.remove(i)
                for param in self.blocks[i].parameters():
                    param.requires_grad_(True)
            return

        # First epoch after warmup - print diagnostics
        #if self.current_epoch == self.warmup_epochs and self.print_diagnostics:
            #print(f"Layer importance estimates:")
            #for i in range(self.num_layers):
                #print(f"Layer {i}: Importance={self.layer_importance[i]:.4f}, Budget={self.beta[i]:.4f}")

        # Gradual freezing schedule - prevent freezing too many layers at once
        epochs_after_warmup = self.current_epoch - self.warmup_epochs
        max_to_freeze = min(self.num_layers - self.n_train_layers,
                            1 + epochs_after_warmup)  # At most 1 more layer per epoch

        # Using improved layer importance for selection, not just budgets
        # This better identifies truly important layers
        sorted_indices = torch.argsort(self.beta)

        # Apply BIAS against freezing early layers (critical for representations)
        early_layer_indices = set(range(3))  # First 3 layers
        biased_indices = []

        # First add indices that are not early layers
        for idx in sorted_indices:
            if idx.item() not in early_layer_indices:
                biased_indices.append(idx.item())

        # Then add early layers (will be frozen last)
        for idx in sorted_indices:
            if idx.item() in early_layer_indices:
                biased_indices.append(idx.item())

        # Get indices of layers to freeze
        to_freeze = biased_indices[:max_to_freeze]

        # Update frozen status for each layer
        new_frozen = set(to_freeze)
        for i in range(self.num_layers):
            is_trainable = i not in new_frozen
            for param in self.blocks[i].parameters():
                param.requires_grad_(is_trainable)

        # Update the frozen_layers set
        self.frozen_layers = new_frozen

        # Print freezing diagnostics
        #if self.print_diagnostics:
        #    print(f"Epoch {self.current_epoch} - Freezing {len(new_frozen)} layers: {sorted(new_frozen)}")


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

    # IMPORTANT: These next lines should only execute AFTER the epoch loop is complete
    # They are currently indented incorrectly in your code
    # Restore best model
    if best_model_state is not None:
        alast_model.load_state_dict(best_model_state)
        print(f"Restored best model with validation accuracy: {best_accuracy:.2f}%")

    # Save final metrics safely
    os.makedirs('results', exist_ok=True)

    try:
        with open('results/alast_metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for k, v in metrics.items():
                if k == 'layer_budgets':
                    serializable_metrics[k] = [b.tolist() if hasattr(b, 'tolist') else b for b in v]
                else:
                    serializable_metrics[k] = [float(x) if isinstance(x, (np.number, torch.Tensor)) else x for x in v]
            json.dump(serializable_metrics, f, indent=2)
        print("ALaST metrics saved successfully to results/alast_metrics.json")
    except Exception as e:
        print(f"Error saving ALaST metrics: {str(e)}")
        import traceback
        traceback.print_exc()

    return alast_model, metrics, best_accuracy


def evaluate(model, dataloader, device, mixed_precision=True):
    """Evaluate model accuracy with support for both traditional and ALaST models"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if mixed_precision and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    # Check if this is an ALaST model or a standard model
                    if hasattr(model, 'update_budgets'):  # It's an ALaST model
                        outputs = model(images, compute_deltas=False)
                    else:  # It's a standard model
                        outputs = model(images)
            else:
                # Check if this is an ALaST model or a standard model
                if hasattr(model, 'update_budgets'):  # It's an ALaST model
                    outputs = model(images, compute_deltas=False)
                else:  # It's a standard model
                    outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


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
        with open('results/traditional_food101_metrics.json', 'w') as f:
            serializable_metrics = {}
            for k, v in metrics.items():
                serializable_metrics[k] = [float(x) if isinstance(x, (np.number, torch.Tensor)) else x for x in v]
            json.dump(serializable_metrics, f, indent=2)
    except Exception as e:
        print(f"Error saving traditional metrics: {str(e)}")
        import traceback
        traceback.print_exc()

    return model, metrics, best_accuracy


def run_comparison():
    """Run both traditional and ALaST training for comparison"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Optional: Use a subset for faster experimentation
    # subset_size = int(0.5 * len(train_dataset))
    # indices = torch.randperm(len(train_dataset))[:subset_size]
    # train_dataset = torch.utils.data.Subset(train_dataset, indices)
    # val_subset_size = int(0.5 * len(val_dataset))
    # val_indices = torch.randperm(len(val_dataset))[:val_subset_size]
    # val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    # Common parameters
    num_epochs = 15
    learning_rate = 1e-4
    mixed_precision = torch.cuda.is_available()

    try:
        # 1. Run traditional fine-tuning
        print("\n===== Running Traditional Fine-tuning =====")
        traditional_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=101)
        traditional_model, trad_metrics, trad_best_acc = train_traditional_vit(
            model=traditional_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            device=device,
            mixed_precision=mixed_precision
        )

        # Save traditional model
        torch.save(traditional_model.state_dict(), 'results/traditional_food101.pth')

        # 2. Run ALaST fine-tuning
        print("\n===== Running Improved ALaST Fine-tuning =====")
        alast_model_base = create_model('vit_base_patch16_224', pretrained=True, num_classes=101)
        alast_model, alast_metrics, alast_best_acc = train_alast(
            model=alast_model_base,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            device=device,
            n_train_layers=9,
            mixed_precision=mixed_precision
        )

        # Save ALaST model
        torch.save(alast_model.state_dict(), 'results/alast_food101.pth')

        # 3. Generate comparison plots
        print("\n===== Generating Comparison =====")
        generate_comparison_plots(trad_metrics, alast_metrics, alast_model)

    except Exception as e:
        print(f"An error occurred during comparison: {str(e)}")
        import traceback
        traceback.print_exc()


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
    plt.savefig('results/traditional_vs_alast_comparison.png')

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
    plt.savefig('results/alast_layer_budgets.png')

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
    plt.savefig('results/alast_active_layers.png')

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
        with open('results/food101_comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)

        print("Detailed comparison metrics saved to results/food101_comparison_results.json")
    except Exception as e:
        print(f"Error saving comparison results: {str(e)}")


def main():
    """Main function to run ALaST training on Food-101"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Data transformations with strong augmentation for Food-101
    transform_train = transforms.Compose([
        transforms.Resize(256),  # Resize to slightly larger
        transforms.RandomCrop(224),  # Then crop to target size
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Color augmentation
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

    # Optional: Use a subset for faster experimentation
    # Comment these lines out for full training
    # subset_size = int(0.2 * len(train_dataset))  # 20% of data for faster testing
    # indices = torch.randperm(len(train_dataset))[:subset_size]
    # train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Prepare dataloaders
    batch_size = 16  # Smaller batch size for Food-101
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataset: Food-101 with {len(train_dataset)} training and {len(val_dataset)} validation images")

    # Create model
    print("Creating model...")
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=101)

    # Training parameters
    num_epochs = 15  # Increased for better convergence on Food-101
    learning_rate = 1e-4
    mixed_precision = torch.cuda.is_available()

    print(f"Starting improved ALaST fine-tuning for {num_epochs} epochs with lr={learning_rate}")

    # Train with ALaST
    alast_model, metrics, best_accuracy = train_alast(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        n_train_layers=9,
        mixed_precision=mixed_precision
    )

    # Save trained model
    try:
        torch.save(alast_model.state_dict(), 'results/alast_food101.pth')
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    # Plot training metrics
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

        # Plot layer budgets
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
        plt.savefig('results/alast_training_metrics.png')
    except Exception as e:
        print(f"Error creating plots: {str(e)}")

    print(f"Training complete!")
    print(f"Final validation accuracy: {metrics['val_acc'][-1]:.2f}%")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Average epoch time: {np.mean(metrics['time_per_epoch']):.2f}s")
    if 'peak_memory' in metrics and metrics['peak_memory']:
        print(f"Average peak memory: {np.mean(metrics['peak_memory']):.2f}GB")
    print(f"Plots saved to results/alast_training_metrics.png")


if __name__ == "__main__":
    # Choose which mode to run:
    main()  # Just run ALaST training
    #run_comparison()  # Run both traditional and ALaST training for comparison
