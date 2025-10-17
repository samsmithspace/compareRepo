import numpy as np
import torch


class ViTAttentionAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device  # Get the device of the model

    def get_attention_weights_vit(self, pixel_values):

        pixel_values = pixel_values.to(self.device)
        ks = []
        blocks = self.model.blocks
        num_layers = len(blocks)

        query_outputs = [None] * num_layers
        key_outputs = [None] * num_layers

        # Define hook function with layer index
        def get_qk_hook(layer_idx):
            def hook(module, input, output):
                B, N, _ = input[0].shape
                qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                query_outputs[layer_idx] = q.detach().cpu()
                key_outputs[layer_idx] = k.detach().cpu()
            return hook

        # Register hooks for all layers
        handles = []
        for i in range(num_layers):
            attention_layer = blocks[i].attn  # Access the attention layer
            hook = get_qk_hook(layer_idx=i)
            handle = attention_layer.register_forward_hook(hook)
            handles.append(handle)

        # Ensure pixel_values have exactly 4 dimensions (batch_size, channels, height, width)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.dim() > 4:
            pixel_values = pixel_values.squeeze()  # Remove extra singleton dimensions

        with torch.no_grad():
            self.model(pixel_values)

        # Remove the hooks
        for handle in handles:
            handle.remove()

        # Process outputs
        for i in range(num_layers):
            query = query_outputs[i]
            key = key_outputs[i]
            if query is None or key is None:
                continue

            # Reshape query and key to separate heads
            batch_size, num_heads, num_tokens, head_dim = query.shape
            # Since q and k tensors are of shape (B, num_heads, N, head_dim)
            # No need to reshape again
            # Compute attention scores and probabilities
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

            # Compute k-values
            sorted_probs, _ = torch.sort(attention_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            exceeds_threshold = cumulative_probs > 0.9
            exceeds_any = exceeds_threshold.any(dim=-1)
            k_indices = torch.argmax(exceeds_threshold.float(), dim=-1)
            k_indices = torch.where(exceeds_any, k_indices, attention_probs.size(-1) - 1)
            k_values = k_indices + 1  # Shape: [B, H, N]

            # Compute median across tokens and batch
            median_last_dim = k_values.median(dim=2).values  # Median over tokens
            median_result = median_last_dim.median(dim=0).values  # Median over batch

            ks.append(median_result.cpu().tolist())

        return ks

