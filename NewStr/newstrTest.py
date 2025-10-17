import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
from collections import deque
import numpy as np
from tqdm import tqdm
import time
import torch.nn.functional as F

# Global dictionary to store attention weights (avoids circular references)
_attention_storage = {}

# ============================================================================
# Layer Instability Index (LII) Tracker
# ============================================================================

class LIITracker:
    """Tracks Layer Instability Index for each transformer layer."""
    
    def __init__(self, num_layers, window_size=20):
        self.num_layers = num_layers
        self.window_size = window_size
        self.buffers = {i: deque(maxlen=window_size) for i in range(num_layers)}
    
    def compute_operational_mode(self, attention_weights):
        """
        Compute k_bar: median minimum number of tokens to accumulate 90% attention mass.
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        
        Returns:
            k_bar: operational mode for this batch
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average over query positions (we care about where attention goes)
        attn = attention_weights.mean(dim=2)  # [batch, heads, seq_len]
        
        # Sort attention scores in descending order
        sorted_attn, _ = torch.sort(attn, dim=-1, descending=True)
        
        # Compute cumulative sum
        cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
        
        # Find minimum k where cumsum >= 0.9
        k_values = (cumsum_attn >= 0.9).float().argmax(dim=-1) + 1
        
        # Return median k across batch and heads
        k_bar = k_values.float().median().item()
        
        return k_bar
    
    def update(self, layer_idx, k_bar):
        """Update buffer for a specific layer."""
        self.buffers[layer_idx].append(k_bar)
    
    def compute_lii(self, layer_idx):
        """
        Compute LII: Median Absolute Deviation of k_bar values.
        
        LII = median(|k_bar_t - median(k_bar)|)
        """
        if len(self.buffers[layer_idx]) < 2:
            return float('inf')  # Not enough data yet
        
        values = np.array(list(self.buffers[layer_idx]))
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        
        return mad
    
    def get_all_liis(self):
        """Get LII for all layers."""
        return {i: self.compute_lii(i) for i in range(self.num_layers)}


# ============================================================================
# Attention Module Wrapper (Fixed - no circular reference)
# ============================================================================

class AttentionWrapper(nn.Module):
    """Wraps the attention module to capture Q, K, V and compute attention."""
    
    def __init__(self, original_attention, num_heads, head_dim, layer_idx, model_id):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.model_id = model_id  # Use ID instead of reference
        
    def forward(self, hidden_states, *args, **kwargs):
        """
        Forward pass that computes attention weights from Q, K, V.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Call the original forward to get the output
        output = self.original_attention(hidden_states, *args, **kwargs)
        
        # Check if we should compute attention (via global flag)
        if _attention_storage.get('compute_flag', False):
            with torch.no_grad():
                try:
                    # Find the qkv projection or separate q, k, v projections
                    attention_module = self.original_attention
                    
                    # Navigate to find the actual attention computation module
                    if hasattr(attention_module, 'attention'):
                        inner_attn = attention_module.attention
                    else:
                        inner_attn = attention_module
                    
                    # Try different structures to get Q, K, V
                    Q = K = V = None
                    
                    # Method 1: Separate query, key, value layers
                    if hasattr(inner_attn, 'query'):
                        Q = inner_attn.query(hidden_states)
                        K = inner_attn.key(hidden_states)
                        V = inner_attn.value(hidden_states)
                    
                    # Method 2: Combined qkv layer
                    elif hasattr(inner_attn, 'qkv'):
                        qkv = inner_attn.qkv(hidden_states)
                        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
                        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
                        Q, K, V = qkv[0], qkv[1], qkv[2]
                    
                    # Method 3: Look for weight matrices directly
                    elif hasattr(inner_attn, 'in_proj_weight'):
                        # PyTorch MultiheadAttention style
                        qkv = F.linear(hidden_states, inner_attn.in_proj_weight, inner_attn.in_proj_bias)
                        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
                        qkv = qkv.permute(2, 0, 3, 1, 4)
                        Q, K, V = qkv[0], qkv[1], qkv[2]
                    
                    if Q is not None:
                        # Reshape if needed
                        if Q.dim() == 3:  # [batch, seq, hidden]
                            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                        
                        # Compute attention: softmax(Q @ K^T / sqrt(d_k))
                        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        
                        # Store in global storage
                        if 'weights' not in _attention_storage:
                            _attention_storage['weights'] = {}
                        _attention_storage['weights'][self.layer_idx] = attn_probs.detach()
                    
                except Exception as e:
                    if self.layer_idx == 0:  # Only print once
                        print(f"[WARNING] Could not compute attention for layer {self.layer_idx}: {e}")
        
        return output


# ============================================================================
# Modified ViT with Attention Patching
# ============================================================================

class ViTWithAttentionTracking(nn.Module):
    """ViT wrapper that patches attention modules to capture Q, K, V."""
    
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=100):
        super().__init__()
        
        # Load pre-trained ViT
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.num_layers = len(self.model.vit.encoder.layer)
        self.model_id = id(self)  # Unique ID for this model
        
        # Get model configuration
        self.config = self.model.config
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Debug: print structure
        self._debug_attention_structure()
        
        # Patch attention modules
        self._patch_attention_modules()
    
    def _debug_attention_structure(self):
        """Debug function to inspect attention module structure."""
        print("\n[DEBUG] Inspecting ViT attention structure:")
        layer = self.model.vit.encoder.layer[0]
        print(f"  Layer type: {type(layer).__name__}")
        
        if hasattr(layer, 'attention'):
            attn = layer.attention
            print(f"  Attention type: {type(attn).__name__}")
            
            if hasattr(attn, 'attention'):
                inner_attn = attn.attention
                print(f"  Inner attention type: {type(inner_attn).__name__}")
                
                # Check for projection layers
                for attr_name in ['query', 'key', 'value', 'qkv', 'in_proj_weight']:
                    if hasattr(inner_attn, attr_name):
                        attr = getattr(inner_attn, attr_name)
                        if isinstance(attr, nn.Module):
                            print(f"  Found {attr_name}: {type(attr).__name__}")
                        elif isinstance(attr, nn.Parameter):
                            print(f"  Found {attr_name}: Parameter {attr.shape}")
        print()
    
    def _patch_attention_modules(self):
        """Replace attention modules with wrapped versions."""
        for layer_idx, layer in enumerate(self.model.vit.encoder.layer):
            if hasattr(layer, 'attention'):
                # Wrap the attention module
                original_attention = layer.attention
                wrapped_attention = AttentionWrapper(
                    original_attention,
                    self.num_heads,
                    self.head_dim,
                    layer_idx,
                    self.model_id
                )
                layer.attention = wrapped_attention
    
    @property
    def attention_weights(self):
        """Get attention weights from global storage."""
        return _attention_storage.get('weights', {})
    
    def forward(self, pixel_values, labels=None, compute_attention=True):
        """Forward pass with attention tracking."""
        # Clear and set global storage
        _attention_storage['weights'] = {}
        _attention_storage['compute_flag'] = compute_attention
        
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels
        )
        
        return outputs
    
    def freeze_layer(self, layer_idx):
        """Freeze a specific transformer layer."""
        layer = self.model.vit.encoder.layer[layer_idx]
        for param in layer.parameters():
            param.requires_grad = False
        print(f"  [FROZEN] Layer {layer_idx}")
    
    def get_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def remove_hooks(self):
        """Remove all attention hooks (no-op for this implementation)."""
        pass


# ============================================================================
# ELA-ViT Training Framework
# ============================================================================

class ELAViTTrainer:
    """Energy Landscape-Aware ViT Training Framework."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-5,
        weight_decay=0.1,
        warmup_steps=60,
        freeze_threshold=2.0,
        window_size=20
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # LII tracking
        self.lii_tracker = LIITracker(
            num_layers=model.num_layers,
            window_size=window_size
        )
        
        # Training parameters
        self.warmup_steps = warmup_steps
        self.freeze_threshold = freeze_threshold
        self.frozen_layers = set()
        
        self.step_count = 0
        self.freeze_decision_made = False
    
    def train_step(self, batch):
        """Single training step."""
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass - only compute attention during warmup
        compute_attention = (self.step_count < self.warmup_steps)
        outputs = self.model(
            pixel_values=images, 
            labels=labels,
            compute_attention=compute_attention
        )
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update LII during warmup
        if self.step_count < self.warmup_steps:
            self._update_lii()
        
        self.step_count += 1
        
        return loss.item()
    
    def _update_lii(self):
        """Update LII for all layers based on captured attention."""
        attention_weights = self.model.attention_weights
        num_captured = len(attention_weights)
        
        for layer_idx in range(self.model.num_layers):
            if layer_idx in attention_weights:
                attn = attention_weights[layer_idx]
                k_bar = self.lii_tracker.compute_operational_mode(attn)
                self.lii_tracker.update(layer_idx, k_bar)
        
        # Debug: Print buffer lengths occasionally
        if self.step_count % 20 == 0:
            buffer_lens = {i: len(self.lii_tracker.buffers[i]) for i in range(self.model.num_layers)}
            print(f"\n[DEBUG] Step {self.step_count}")
            print(f"[DEBUG] Attention weights captured: {num_captured}/{self.model.num_layers} layers")
            print(f"[DEBUG] Buffer lengths: {buffer_lens}")
            # Print a sample k_bar value
            if 0 in attention_weights:
                sample_attn = attention_weights[0]
                sample_k = self.lii_tracker.compute_operational_mode(sample_attn)
                print(f"[DEBUG] Sample k_bar for layer 0: {sample_k:.2f}")
                print(f"[DEBUG] Attention shape: {sample_attn.shape}")
    
    def make_freeze_decision(self):
        """Decide which layers to freeze based on LII."""
        if self.freeze_decision_made:
            return
        
        print("\n" + "="*60)
        print("MAKING FREEZE DECISION")
        print("="*60)
        
        liis = self.lii_tracker.get_all_liis()
        
        print(f"\nLayer Instability Index (LII) values:")
        for layer_idx, lii in liis.items():
            status = "STABLE" if lii < self.freeze_threshold else "ADAPTIVE"
            print(f"  Layer {layer_idx:2d}: LII = {lii:6.3f} [{status}]")
        
        # Freeze layers below threshold
        print(f"\nFreezing layers with LII < {self.freeze_threshold}:")
        frozen_count = 0
        for layer_idx, lii in liis.items():
            if lii < self.freeze_threshold and lii != float('inf'):
                self.model.freeze_layer(layer_idx)
                self.frozen_layers.add(layer_idx)
                frozen_count += 1
        
        if frozen_count == 0:
            print("  No layers frozen (all LII values above threshold or insufficient data)")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.model.get_trainable_params()
        frozen_pct = 100 * (1 - trainable_params / total_params)
        
        print(f"\n{'='*60}")
        print(f"Frozen layers: {sorted(self.frozen_layers)}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen: {frozen_pct:.1f}%")
        print(f"{'='*60}\n")
        
        self.freeze_decision_made = True
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Make freeze decision after warmup
            if self.step_count == self.warmup_steps:
                self.make_freeze_decision()
            
            # Compute accuracy
            images, labels = batch
            images = images.to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values=images, compute_attention=False)
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions.cpu() == labels).sum().item()
                total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{100*correct/total:.2f}%',
                'step': self.step_count
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """Validate on test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(pixel_values=images, compute_attention=False)
            predictions = outputs.logits.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        return accuracy
    
    def train(self, num_epochs):
        """Full training loop."""
        print("="*60)
        print("STARTING ELA-ViT TRAINING")
        print("="*60)
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Freeze threshold: {self.freeze_threshold}")
        print(f"Number of epochs: {num_epochs}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_acc = self.validate()
            
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc:  {train_acc:.2f}%")
            print(f"  Val Acc:    {val_acc:.2f}%")
            print()
        
        total_time = time.time() - start_time
        
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Final validation accuracy: {val_acc:.2f}%")
        print("="*60)
        
        return val_acc


# ============================================================================
# Data Loading
# ============================================================================

def get_cifar100_loaders(batch_size=32, num_workers=1):
    """Get CIFAR-100 data loaders with appropriate transforms."""
    
    # Transforms for ViT (resize to 224x224)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'model_name': 'google/vit-base-patch16-224',
        'num_classes': 100,
        'batch_size': 32,
        'num_epochs': 10,
        'lr': 1e-5,
        'weight_decay': 0.1,
        'warmup_steps': 60,
        'freeze_threshold': 2.0,
        'window_size': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader = get_cifar100_loaders(
        batch_size=config['batch_size']
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print()
    
    # Create model
    print("Loading ViT model...")
    model = ViTWithAttentionTracking(
        model_name=config['model_name'],
        num_classes=config['num_classes']
    )
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create trainer
    trainer = ELAViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        freeze_threshold=config['freeze_threshold'],
        window_size=config['window_size']
    )
    
    # Train
    final_accuracy = trainer.train(num_epochs=config['num_epochs'])
    
    # Cleanup
    model.remove_hooks()
    
    return final_accuracy


if __name__ == '__main__':
    main()