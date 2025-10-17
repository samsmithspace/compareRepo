import torch
import torch.nn as nn
import torch.nn.functional as F

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