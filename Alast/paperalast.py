#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact implementation of ALaST (Adaptive Layer Selection Fine-Tuning)
as described in the paper by Devoto et al. 2024.

Reference: "Adaptive Layer Selection for Efficient Vision Transformer Fine-Tuning"
arXiv:2408.08670v1 [cs.CV] 16 Aug 2024

Key implementation details matching the paper:
1. Budget update using Equation 6: b[l]^(i+1) = b[l]^i + (Δ[l]^(i+1) - Δ[l]^i) × α
2. CLS token delta: Δ[l] = (CLS[l] - CLS[l-1])²
3. Probabilistic layer sampling from budget distribution
4. Token selection based on CLS attention scores
5. Per-batch budget updates
"""

import math
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Vision Transformer Components
# -------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_cls_attention_scores(self, x):
        """
        Compute attention scores from CLS token to all tokens.
        Returns: (B, N) attention scores
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Get CLS query (first token)
        q_cls = q[:, :, 0:1, :]  # (B, num_heads, 1, head_dim)

        # Compute attention from CLS to all tokens
        attn = (q_cls @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, N)
        attn = F.softmax(attn, dim=-1)  # (B, num_heads, 1, N)

        # Average over heads and squeeze
        cls_scores = attn.mean(dim=1).squeeze(1)  # (B, N)

        return cls_scores

    def forward(self, x, keep_indices=None):
        """
        x: (B, N, C)
        keep_indices: (B, S) - indices of tokens to keep, if None process all
        """
        B, N, C = x.shape

        if keep_indices is not None:
            # Gather tokens to keep
            idx_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, S, C)
            x = torch.gather(x, dim=1, index=idx_expanded)  # (B, S, C)
            N = x.shape[1]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, budget, return_cls_delta=False):
        """
        x: (B, N, C) where first token is CLS
        budget: float in [0, 1] - percentage of tokens to keep
        return_cls_delta: if True, return the CLS token change
        """
        B, N, C = x.shape
        cls_in = x[:, 0, :].clone()  # Store input CLS for delta calculation

        # Normalize
        x_norm = self.norm1(x)

        # Token selection based on CLS attention (Paper section 3.2)
        if budget < 1.0:
            # Get CLS attention scores
            cls_scores = self.attn.get_cls_attention_scores(x_norm)  # (B, N)

            # Exclude CLS token from ranking (always keep it)
            non_cls_scores = cls_scores[:, 1:]  # (B, N-1)

            # Keep top budget% of tokens
            keep_k = max(1, int(budget * (N - 1)))
            topk_vals, topk_idx = torch.topk(non_cls_scores, k=keep_k, dim=1)  # (B, keep_k)

            # Adjust indices (add 1 because we excluded CLS)
            topk_idx = topk_idx + 1

            # Always include CLS token (index 0)
            cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            keep_indices = torch.cat([cls_idx, topk_idx], dim=1)  # (B, keep_k+1)

            # Gather tokens to keep
            idx_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, C)
            x_kept = torch.gather(x, dim=1, index=idx_expanded)
            x_norm_kept = torch.gather(x_norm, dim=1, index=idx_expanded)
        else:
            x_kept = x
            x_norm_kept = x_norm
            keep_indices = None

        # Self-attention
        attn_out = self.attn(x_norm_kept, keep_indices=None)  # Already selected tokens

        # Residual connection
        x_out = x_kept + attn_out

        # FFN
        x_out = x_out + self.mlp(self.norm2(x_out))

        if return_cls_delta:
            cls_out = x_out[:, 0, :]  # Output CLS
            # Equation 5 from paper: Δ[l] = (CLS[l] - CLS[l-1])²
            cls_delta_squared = ((cls_out - cls_in) ** 2).sum(dim=1).mean()  # Scalar
            return x_out, cls_delta_squared

        return x_out, None


class ALaSTViT(nn.Module):
    """
    Vision Transformer with ALaST (Adaptive Layer Selection Fine-Tuning).

    Implements the exact algorithm from the paper:
    - Budget update: b[l]^(i+1) = b[l]^i + (Δ[l]^(i+1) - Δ[l]^i) × α
    - Probabilistic layer selection based on budget distribution
    - Token pruning based on CLS attention
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=100,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            K=9,  # Number of layers to train per iteration (paper uses 9)
            alpha=0.0001,  # Budget learning rate (paper: matches fine-tuning LR)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.K = K  # Number of trainable layers per iteration
        self.alpha = alpha  # Budget learning rate

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # ALaST-specific: Budget and delta tracking
        # Initialize budgets to 1.0 (all tokens, all layers trained)
        self.register_buffer('budgets', torch.ones(depth))
        self.register_buffer('prev_deltas', torch.zeros(depth))
        self.register_buffer('iteration', torch.tensor(0))

        # Track which layers are currently frozen
        self.frozen_layers = set()

    def update_budgets_and_select_layers(self, current_deltas):
        """
        Implements Equation 6 from paper:
        b[l]^(i+1) = b[l]^i + (Δ[l]^(i+1) - Δ[l]^i) × α

        Then samples K layers to train based on budget distribution.
        """
        with torch.no_grad():
            # Update budgets using paper's equation
            delta_change = current_deltas - self.prev_deltas
            self.budgets = self.budgets + delta_change * self.alpha

            # Clip budgets to [0, 1]
            self.budgets = torch.clamp(self.budgets, 0.0, 1.0)

            # Store current deltas for next iteration
            self.prev_deltas = current_deltas.clone()

            # Sample K layers to train based on budget distribution (Paper section 3.2)
            # "We sample layers x ~ D[L] without replacement"
            # Use budgets as sampling weights
            probs = self.budgets / self.budgets.sum()

            try:
                # Sample K layers without replacement
                selected_layers = torch.multinomial(probs, self.K, replacement=False)
                selected_layers = selected_layers.cpu().tolist()
            except RuntimeError:
                # If sampling fails (e.g., all budgets near 0), select top-K
                selected_layers = torch.topk(self.budgets, self.K)[1].cpu().tolist()

            # Update frozen layers
            all_layers = set(range(self.depth))
            self.frozen_layers = all_layers - set(selected_layers)

            # Freeze/unfreeze layers
            for layer_idx in range(self.depth):
                requires_grad = layer_idx not in self.frozen_layers
                for param in self.blocks[layer_idx].parameters():
                    param.requires_grad = requires_grad

            self.iteration += 1

    def forward(self, x, compute_deltas=False):
        """
        x: (B, C, H, W)
        compute_deltas: if True, return CLS deltas for budget update
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Forward through transformer blocks
        deltas = []
        for layer_idx, block in enumerate(self.blocks):
            budget = self.budgets[layer_idx].item()
            x, delta = block(x, budget, return_cls_delta=compute_deltas)
            if compute_deltas:
                deltas.append(delta)

        # Classification head
        cls_token = x[:, 0]  # (B, embed_dim)
        cls_token = self.norm(cls_token)
        logits = self.head(cls_token)

        if compute_deltas:
            deltas_tensor = torch.stack(deltas)  # (depth,)
            return logits, deltas_tensor

        return logits

    def get_budget_info(self):
        """Return current budget allocation for monitoring."""
        return {
            'budgets': self.budgets.cpu().numpy(),
            'frozen_layers': sorted(list(self.frozen_layers)),
            'trainable_layers': sorted(list(set(range(self.depth)) - self.frozen_layers)),
            'iteration': self.iteration.item()
        }


# -------------------------
# Data Loading
# -------------------------

def get_cifar100_loaders(batch_size=128, num_workers=2):
    """Paper uses CIFAR-100 with standard augmentation."""

    # Paper mentions standard augmentation
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ViT expects 224x224
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

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# -------------------------
# Training & Evaluation
# -------------------------

def evaluate(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, compute_deltas=False)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def train_alast(
        model,
        train_loader,
        test_loader,
        num_epochs=20,
        lr=0.0001,
        device='cuda',
        print_freq=100
):
    """
    Train model using ALaST algorithm from paper.

    Paper parameters (from Table 2 & 3):
    - Optimizer: Adam
    - Learning rate: 0.0001
    - Batch size: 128
    - Epochs: 20 (based on convergence plots)
    - K (trainable layers): 9
    - α (budget learning rate): matches fine-tuning LR (0.0001)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters (will change dynamically)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Always keep head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    print(f"Starting ALaST training on {device}")
    print(f"Epochs: {num_epochs}, LR: {lr}, Batch size: {train_loader.batch_size}")
    print(f"K (trainable layers): {model.K}, α (budget LR): {model.alpha}")
    print("-" * 80)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
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

            if (batch_idx + 1) % print_freq == 0:
                budget_info = model.get_budget_info()
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | "
                      f"Frozen layers: {len(budget_info['frozen_layers'])}/{model.depth}")

        # Epoch statistics
        epoch_time = time.time() - start_time
        train_acc = 100.0 * correct / total
        avg_loss = epoch_loss / len(train_loader)

        # Evaluation
        test_acc = evaluate(model, test_loader, device)

        # Budget info
        budget_info = model.get_budget_info()

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Budgets: {np.array2string(budget_info['budgets'], precision=3, suppress_small=True)}")
        print(f"  Trainable layers: {budget_info['trainable_layers']}")
        print(f"  Frozen layers: {budget_info['frozen_layers']}")
        print("-" * 80)

    return model


# -------------------------
# Main
# -------------------------

def main():
    seed_everything(42)
    device = get_device()

    print("=" * 80)
    print("ALaST: Adaptive Layer Selection Fine-Tuning")
    print("Exact implementation following Devoto et al. 2024")
    print("=" * 80)

    # Paper hyperparameters (from experimental setup and tables)
    num_epochs = 20
    batch_size = 128
    lr = 0.0001
    K = 9  # Number of trainable layers

    # Model configuration: DeiT-S from paper (Table 1)
    # embed_dim=384, num_heads=6, depth=12
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

    # Load data
    print("\nLoading CIFAR-100...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size, num_workers=2)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable initially")

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    model = train_alast(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        print_freq=100
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training complete! Final evaluation:")
    final_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    budget_info = model.get_budget_info()
    print(f"\nFinal budget allocation:")
    for i, budget in enumerate(budget_info['budgets']):
        status = "FROZEN" if i in budget_info['frozen_layers'] else "ACTIVE"
        print(f"  Layer {i:2d}: budget={budget:.3f} [{status}]")

    print("=" * 80)


if __name__ == "__main__":
    main()