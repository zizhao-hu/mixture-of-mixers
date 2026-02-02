# Mixture of Mixers (MoM)
# Attention replacement using MoE-style token mixers
#
# Architecture:
#   - Token Mixers: 2-layer MLP mixing across the token/spatial dimension (N)
#   - Normalization along token dimension before mixing
#   - Router selects top-k experts
#   - Output projection after weighted sum

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp

from .common import (
    modulate,
    TimestepEmbedder,
    LabelEmbedder,
    FinalLayer,
    get_2d_sincos_pos_embed
)


#################################################################################
#                              Token Mixer Expert                                #
#################################################################################

class TokenMixer(nn.Module):
    """
    Token Mixer: 2-layer MLP mixing across token/spatial dimension.
    
    Input shape: (B, N, D)
    Operation: 
        1. Normalize along token dim (dim=1)
        2. Transpose to (B, D, N)
        3. MLP: N -> hidden -> N
        4. Transpose back to (B, N, D)
    """
    def __init__(self, num_tokens, hidden_size, hidden_ratio=1.0):
        super().__init__()
        hidden_dim = int(num_tokens * hidden_ratio)
        self.fc1 = nn.Linear(num_tokens, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, num_tokens)
    
    def forward(self, x):
        # x: (B, N, D)
        # Normalize along token dimension
        x = F.layer_norm(x.transpose(1, 2), [x.size(1)]).transpose(1, 2)
        
        x = x.transpose(1, 2)  # (B, D, N)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadBatchedMixers(nn.Module):
    """
    Multi-Head Unified Mixers.
    
    Each expert application performs both Token Mixing (across tokens N) 
    and Channel Mixing (across head dimension hd) in a two-layer structure.
    
    Layer 1: h = GELU(W1_chan @ x @ W1_tok^T + B1)
    Layer 2: out = W2_chan @ h @ W2_tok^T + B2
    """
    def __init__(self, num_experts, num_heads, num_tokens, head_dim, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        
        # Intermediate dimensions: 
        # Channels: head_dim (hd) -> head_dim (hd)
        # Tokens: num_tokens (N) -> hidden_dim (E_dim)
        
        # Layer 1 Weights
        self.fc1_tok_weight = nn.Parameter(torch.empty(num_experts, num_heads, hidden_dim, num_tokens))
        self.fc1_chan_weight = nn.Parameter(torch.empty(num_experts, num_heads, head_dim, head_dim))
        self.fc1_bias = nn.Parameter(torch.zeros(num_experts, num_heads, head_dim, hidden_dim))
        
        # Layer 2 Weights
        self.fc2_tok_weight = nn.Parameter(torch.empty(num_experts, num_heads, num_tokens, hidden_dim))
        self.fc2_chan_weight = nn.Parameter(torch.empty(num_experts, num_heads, head_dim, head_dim))
        self.fc2_bias = nn.Parameter(torch.zeros(num_experts, num_heads, head_dim, num_tokens))
        
        # Initialize
        for i in range(num_experts):
            for h in range(num_heads):
                nn.init.kaiming_uniform_(self.fc1_tok_weight[i, h], a=5**0.5)
                nn.init.kaiming_uniform_(self.fc1_chan_weight[i, h], a=5**0.5)
                nn.init.kaiming_uniform_(self.fc2_tok_weight[i, h], a=5**0.5)
                nn.init.kaiming_uniform_(self.fc2_chan_weight[i, h], a=5**0.5)
    
    def forward(self, x, expert_indices, expert_weights):
        """
        Args:
            x: (B, H, hd, N) input tensor
            expert_indices: (B, H, top_k)
            expert_weights: (B, H, top_k)
        """
        B, H, hd, N = x.shape
        top_k = expert_indices.shape[2]
        E_dim = self.hidden_dim
        
        flat_expert_indices = expert_indices.reshape(-1)
        head_indices = torch.arange(H, device=x.device).view(1, H, 1).expand(B, H, top_k).reshape(-1)
        
        # Gather weights
        w1_t = self.fc1_tok_weight[flat_expert_indices, head_indices] # (BHK, E_dim, N)
        w1_c = self.fc1_chan_weight[flat_expert_indices, head_indices] # (BHK, hd, hd)
        b1 = self.fc1_bias[flat_expert_indices, head_indices]       # (BHK, hd, E_dim)
        
        w2_t = self.fc2_tok_weight[flat_expert_indices, head_indices] # (BHK, N, E_dim)
        w2_c = self.fc2_chan_weight[flat_expert_indices, head_indices] # (BHK, hd, hd)
        b2 = self.fc2_bias[flat_expert_indices, head_indices]       # (BHK, hd, N)
        
        # Expand input: (BHK, hd, N)
        x_exp = x.unsqueeze(2).expand(-1, -1, top_k, -1, -1).reshape(B * H * top_k, hd, N)
        
        # Layer 1: h = GELU(W1_c @ x @ W1_t.T + B1)
        # (BHK, hd, N) @ (BHK, N, E_dim) -> (BHK, hd, E_dim)
        h = torch.bmm(x_exp, w1_t.transpose(-1, -2))
        # (BHK, hd, hd) @ (BHK, hd, E_dim) -> (BHK, hd, E_dim)
        h = torch.bmm(w1_c, h) + b1
        h = F.gelu(h, approximate="tanh")
        
        # Layer 2: out = W2_c @ h @ W2_t.T + B2
        # (BHK, hd, E_dim) @ (BHK, E_dim, N) -> (BHK, hd, N)
        out = torch.bmm(h, w2_t.transpose(-1, -2))
        # (BHK, hd, hd) @ (BHK, hd, N) -> (BHK, hd, N)
        out = torch.bmm(w2_c, out) + b2
        
        # Reshape and aggregate
        out = out.reshape(B, H, top_k, hd, N)
        weights = expert_weights.view(B, H, top_k, 1, 1)
        out = (out * weights).sum(dim=2)  # (B, H, hd, N)
        
        return out



class MultiHeadLinearBatchedTokenMixers(nn.Module):
    """
    Multi-Head Linear Batched Token Mixers.
    
    Uses a single learned N x N transition matrix per expert head.
    Softmax is applied to the rows of the weight matrix before operation.
    """
    def __init__(self, num_experts, num_heads, num_tokens):
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        
        # Expert weights: (E, H, N, N)
        self.weight = nn.Parameter(torch.empty(num_experts, num_heads, num_tokens, num_tokens))
        self.bias = nn.Parameter(torch.zeros(num_experts, num_heads, num_tokens))
        
        # Initialize
        for i in range(num_experts):
            for h in range(num_heads):
                nn.init.kaiming_uniform_(self.weight[i, h], a=5**0.5)
    
    def forward(self, x, expert_indices, expert_weights):
        """
        Args:
            x: (B, H, head_dim, N) input tensor
            expert_indices: (B, H, top_k) indices
            expert_weights: (B, H, top_k) weights
        """
        B, H, hd, N = x.shape
        top_k = expert_indices.shape[2]
        
        flat_expert_indices = expert_indices.reshape(-1)
        head_indices = torch.arange(H, device=x.device).view(1, H, 1).expand(B, H, top_k).reshape(-1)
        
        # Gather weights: (B*H*top_k, N, N)
        w = self.weight[flat_expert_indices, head_indices] # (B*H*top_k, N, N)
        b = self.bias[flat_expert_indices, head_indices]     # (B*H*top_k, N)
        
        # Softmax on rows: each output token is a convex combination of input tokens
        w_attn = F.softmax(w, dim=-1)
        
        # Expand input: (B, H, hd, N) -> (B*H*top_k, hd, N)
        x_expanded = x.unsqueeze(2).expand(-1, -1, top_k, -1, -1).reshape(B * H * top_k, hd, N)
        
        # Apply transition: (B*H*top_k, hd, N) @ (B*H*top_k, N, N)^T -> (B*H*top_k, hd, N)
        out = torch.bmm(x_expanded, w_attn.transpose(-1, -2))
        out = out + b.unsqueeze(1)
        
        # Reshape and aggregate: (B*H*top_k, hd, N) -> (B, H, top_k, hd, N)
        out = out.reshape(B, H, top_k, hd, N)
        weights = expert_weights.view(B, H, top_k, 1, 1)
        out = (out * weights).sum(dim=2)  # (B, H, hd, N)
        
        return out


#################################################################################
#                    Mixture of Mixers (Attention Replacement)                   #
#################################################################################

class MixtureOfMixers(nn.Module):
    """
    Mixture of Mixers: Drop-in attention replacement using Multi-Head MoE token mixers.
    
    Args:
        hidden_size: Feature dimension (D)
        num_tokens: Number of tokens (N)
        num_experts: Total number of experts (default: 8)
        top_k: Number of experts to select (default: 2)
        num_heads: Number of heads for the multi-head expertise (default: 8)
        hidden_ratio: Hidden dimension expansion ratio for experts (default: 1.0)
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_experts=8,
        top_k=2,
        num_heads=8,
        hidden_ratio=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_heads = num_heads
        
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.head_dim = hidden_size // num_heads
        
        # Auto-compute hidden_dim (MLP dim per head)
        # Target Param count = 2D² (half of attention's 4D²)
        # Active params = num_heads * top_k * 2 * N * hidden_dim
        # hidden_dim = 2D² / (num_heads * top_k * 2 * N)
        D, N = hidden_size, num_tokens
        calc_hidden_dim = int(2 * D**2 / (num_heads * top_k * 2 * N))
        hidden_dim = int(calc_hidden_dim * hidden_ratio)
        hidden_dim = max(hidden_dim, 1)  # At least 1
        self.hidden_dim = hidden_dim
        
        # Unified mixer experts (Multi-head)
        self.experts = MultiHeadBatchedMixers(
            num_experts=num_experts,
            num_heads=num_heads,
            num_tokens=num_tokens,
            head_dim=self.head_dim,
            hidden_dim=hidden_dim,
        )
        
        # Single Router for all heads
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Input projection (D -> D)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, N, D)
        
        Returns:
            output: Tensor (B, N, D)
            aux_loss: Load balancing loss
        """
        B, N, D = x.shape
        H = self.num_heads
        hd = self.head_dim
        
        # Compute routing scores once for the entire token representation
        router_input = x.mean(dim=1)  # (B, D)
        router_logits = self.router(router_input)  # (B, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts (B, top_k)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Expand routing decisions for each head: (B, top_k) -> (B, H, top_k)
        head_top_k_indices = top_k_indices.unsqueeze(1).expand(-1, H, -1)
        head_top_k_weights = top_k_weights.unsqueeze(1).expand(-1, H, -1)
        
        # Apply input projection
        x_projected = self.in_proj(x)
        
        # Reshape projected input to multi-head format: (B, N, H, hd) -> (B, H, hd, N)
        x_reshaped = x_projected.view(B, N, H, hd).permute(0, 2, 3, 1) # (B, H, hd, N)
        
        # LayerNorm along token dimension (N) for each head/channel
        x_normed = F.layer_norm(x_reshaped, [N]) # (B, H, hd, N)
        
        # Multi-head Expert computation
        output = self.experts(x_normed, head_top_k_indices, head_top_k_weights)  # (B, H, hd, N)
        
        # Reshape back to (B, N, D)
        output = output.permute(0, 3, 1, 2).reshape(B, N, D)
        
        # Output projection
        output = self.out_proj(output)
        
        # Load balancing loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _load_balancing_loss(self, router_probs, top_k_indices):
        """Standard load balancing loss."""
        # router_probs: (B, num_experts)
        # top_k_indices: (B, top_k)
        E = self.num_experts
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=E).float()
        
        expert_probs = router_probs.mean(dim=0)
        expert_fraction = expert_mask.mean(dim=0)
        return self.num_experts * (expert_probs * expert_fraction).sum()


class LinearMixtureOfMixers(nn.Module):
    """
    Linear Mixture of Mixers: Uses Multi-Head Linear (N x N) experts with row-wise softmax.
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_experts=8,
        top_k=2,
        num_heads=8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_heads = num_heads
        
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.head_dim = hidden_size // num_heads
        
        # Linear experts (N x N)
        self.experts = MultiHeadLinearBatchedTokenMixers(
            num_experts=num_experts,
            num_heads=num_heads,
            num_tokens=num_tokens,
        )
        
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Input projection (D -> D)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        hd = self.head_dim
        
        # Compute routing scores once for the entire token representation
        router_input = x.mean(dim=1)  # (B, D)
        router_logits = self.router(router_input)  # (B, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts (B, top_k)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Expand routing decisions for each head: (B, top_k) -> (B, H, top_k)
        head_top_k_indices = top_k_indices.unsqueeze(1).expand(-1, H, -1)
        head_top_k_weights = top_k_weights.unsqueeze(1).expand(-1, H, -1)
        
        # Apply input projection
        x_projected = self.in_proj(x)
        
        # Reshape projected input to multi-head format: (B, N, H, hd) -> (B, H, hd, N)
        x_reshaped = x_projected.view(B, N, H, hd).permute(0, 2, 3, 1) # (B, H, hd, N)
        x_normed = F.layer_norm(x_reshaped, [N]) # (B, H, hd, N)
        
        output = self.experts(x_normed, head_top_k_indices, head_top_k_weights)
        output = output.permute(0, 3, 1, 2).reshape(B, N, D)
        output = self.out_proj(output)
        
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)
        return output, aux_loss

    def _load_balancing_loss(self, router_probs, top_k_indices):
        E = self.num_experts
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=E).float()
        expert_probs = router_probs.mean(dim=0)
        expert_fraction = expert_mask.mean(dim=0)
        return self.num_experts * (expert_probs * expert_fraction).sum()


class SoftMixtureOfMixers(nn.Module):
    """
    Soft Mixture of Mixers: Fully differentiable soft MoE with slot bottleneck.
    
    Key insight: Compute sparsity comes from the slot bottleneck, not from
    turning off experts. Each expert processes S slots (S << N tokens).
    
    1. Dispatch: Compress N tokens → E×S slots via soft dispatch weights
    2. Process: Each expert processes only S slots (not N tokens!) → O(E×S×H)
    3. Combine: Expand E×S slots → N tokens via soft combine weights
    
    This is fully differentiable with compute cost O(S×H) instead of O(N×H).
    
    Args:
        hidden_size: Feature dimension (D)
        num_tokens: Number of tokens (N)
        num_experts: Number of experts (default: 8)
        slots_per_expert: Slots per expert (default: 1, total slots = E×S)
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_experts=8,
        slots_per_expert=1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.total_slots = num_experts * slots_per_expert
        
        # Expert MLP processes slots, not tokens
        # Each expert: D -> hidden_dim -> D (feature-wise MLP on each slot)
        D = hidden_size
        hidden_dim = D  # Reduced from 4×D for faster speed (similar to MoM)
        self.hidden_dim = hidden_dim
        
        # Per-expert weights: (E, hidden_dim, D) for FC1, (E, D, hidden_dim) for FC2
        self.fc1_weight = nn.Parameter(torch.empty(num_experts, hidden_dim, D))
        self.fc1_bias = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.fc2_weight = nn.Parameter(torch.empty(num_experts, D, hidden_dim))
        self.fc2_bias = nn.Parameter(torch.zeros(num_experts, D))
        
        # Initialize
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.fc1_weight[i], a=5**0.5)
            nn.init.kaiming_uniform_(self.fc2_weight[i], a=5**0.5)
        
        # Dispatch projection: maps each token to logits over all slots
        # phi: (D) -> (E * S) per token
        self.phi = nn.Linear(hidden_size, self.total_slots, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, N, D)
        
        Returns:
            output: Tensor (B, N, D)
            aux_loss: Always 0 (no load balancing needed for soft MoE)
        """
        B, N, D = x.shape
        E = self.num_experts
        S = self.slots_per_expert
        H = self.hidden_dim
        
        # Compute dispatch logits: (B, N, D) -> (B, N, E*S)
        dispatch_logits = self.phi(x)  # (B, N, E*S)
        
        # Dispatch weights: softmax over tokens for each slot
        # This creates a weighted average of tokens for each slot
        dispatch_weights = F.softmax(dispatch_logits, dim=1)  # (B, N, E*S)
        
        # Combine weights: softmax over slots for each token  
        # This determines how to combine slot outputs back to tokens
        combine_weights = F.softmax(dispatch_logits, dim=2)  # (B, N, E*S)
        
        # Dispatch: compress N tokens -> E*S slots
        # x: (B, N, D), dispatch_weights: (B, N, E*S)
        # slots = dispatch_weights^T @ x: (B, E*S, D)
        slots = torch.einsum("bns,bnd->bsd", dispatch_weights, x)  # (B, E*S, D)
        
        # Reshape slots for per-expert processing: (B, E*S, D) -> (B, E, S, D)
        slots = slots.view(B, E, S, D)
        
        # Process slots through expert MLPs (all experts in parallel)
        # slots: (B, E, S, D) -> reshape -> (B*E, S, D)
        slots = slots.reshape(B * E, S, D)
        
        # FC1: (B*E, S, D) @ (B*E, D, H) -> (B*E, S, H)
        # Expand weights: (E, H, D) -> (B*E, H, D)
        fc1_w = self.fc1_weight.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * E, H, D)
        fc1_b = self.fc1_bias.unsqueeze(0).expand(B, -1, -1).reshape(B * E, H)
        
        # bmm: (B*E, S, D) @ (B*E, D, H) -> (B*E, S, H)
        h = torch.bmm(slots, fc1_w.transpose(-1, -2))  # (B*E, S, H)
        h = h + fc1_b.unsqueeze(1)  # broadcast bias to S dimension
        h = F.gelu(h, approximate="tanh")
        
        # FC2: (B*E, S, H) @ (B*E, H, D) -> (B*E, S, D)
        fc2_w = self.fc2_weight.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * E, D, H)
        fc2_b = self.fc2_bias.unsqueeze(0).expand(B, -1, -1).reshape(B * E, D)
        
        out = torch.bmm(h, fc2_w.transpose(-1, -2))  # (B*E, S, D)
        out = out + fc2_b.unsqueeze(1)
        
        # Reshape back: (B*E, S, D) -> (B, E, S, D) -> (B, E*S, D)
        out = out.view(B, E, S, D).reshape(B, E * S, D)
        
        # Combine: expand E*S slots -> N tokens
        # out: (B, E*S, D), combine_weights: (B, N, E*S)  
        # output = combine_weights @ out: (B, N, D)
        output = torch.einsum("bns,bsd->bnd", combine_weights, out)  # (B, N, D)
        
        # Output projection
        output = self.out_proj(output)
        
        # No aux loss for soft MoE (implicit load balancing via soft weights)
        return output, torch.tensor(0.0, device=x.device)

#################################################################################
#                         DiT-MoM Block (for Diffusion)                          #
#################################################################################

class DiT_MoMBlock(nn.Module):
    """DiT block with MoM replacing attention."""
    def __init__(
        self,
        hidden_size,
        num_tokens,
        hidden_ratio=1.0,
        mlp_ratio=4.0,
        num_heads=16,
    ):
        super().__init__()
        self.mom = MixtureOfMixers(
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            hidden_ratio=hidden_ratio,
            num_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_mom, scale_mom, gate_mom, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        modulated_x = x * (1 + scale_mom.unsqueeze(1)) + shift_mom.unsqueeze(1)
        mom_out, aux_loss = self.mom(modulated_x)
        x = x + gate_mom.unsqueeze(1) * mom_out
        
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x, aux_loss


#################################################################################
#                         DiT-MoM Model (Diffusion Transformer)                  #
#################################################################################

class DiT_MoM(nn.Module):
    """Diffusion Transformer with Mixture of Mixers."""
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        num_heads=16,
        depth=28,
        hidden_ratio=1.0,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.depth = depth
        self.aux_loss_weight = aux_loss_weight

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiT_MoMBlock(
                hidden_size=hidden_size,
                num_tokens=num_patches,
                hidden_ratio=hidden_ratio,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, c)
            total_aux_loss = total_aux_loss + aux_loss
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        return x, total_aux_loss / self.depth

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, _ = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                              DiT-MoM Configs                                   #
#################################################################################

def DiT_MoM_XL_2(**kwargs):
    return DiT_MoM(depth=28, hidden_size=1152, patch_size=2, **kwargs)

def DiT_MoM_XL_4(**kwargs):
    return DiT_MoM(depth=28, hidden_size=1152, patch_size=4, **kwargs)

def DiT_MoM_XL_8(**kwargs):
    return DiT_MoM(depth=28, hidden_size=1152, patch_size=8, **kwargs)

def DiT_MoM_L_2(**kwargs):
    return DiT_MoM(depth=24, hidden_size=1024, patch_size=2, **kwargs)

def DiT_MoM_L_4(**kwargs):
    return DiT_MoM(depth=24, hidden_size=1024, patch_size=4, **kwargs)

def DiT_MoM_L_8(**kwargs):
    return DiT_MoM(depth=24, hidden_size=1024, patch_size=8, **kwargs)

def DiT_MoM_B_2(**kwargs):
    return DiT_MoM(depth=12, hidden_size=768, patch_size=2, **kwargs)

def DiT_MoM_B_4(**kwargs):
    return DiT_MoM(depth=12, hidden_size=768, patch_size=4, **kwargs)

def DiT_MoM_B_8(**kwargs):
    return DiT_MoM(depth=12, hidden_size=768, patch_size=8, **kwargs)

def DiT_MoM_S_2(**kwargs):
    return DiT_MoM(depth=12, hidden_size=384, patch_size=2, **kwargs)

def DiT_MoM_S_4(**kwargs):
    return DiT_MoM(depth=12, hidden_size=384, patch_size=4, **kwargs)

def DiT_MoM_S_8(**kwargs):
    return DiT_MoM(depth=12, hidden_size=384, patch_size=8, **kwargs)


DiT_MoM_models = {
    'DiT-MoM-XL/2': DiT_MoM_XL_2,  'DiT-MoM-XL/4': DiT_MoM_XL_4,  'DiT-MoM-XL/8': DiT_MoM_XL_8,
    'DiT-MoM-L/2':  DiT_MoM_L_2,   'DiT-MoM-L/4':  DiT_MoM_L_4,   'DiT-MoM-L/8':  DiT_MoM_L_8,
    'DiT-MoM-B/2':  DiT_MoM_B_2,   'DiT-MoM-B/4':  DiT_MoM_B_4,   'DiT-MoM-B/8':  DiT_MoM_B_8,
    'DiT-MoM-S/2':  DiT_MoM_S_2,   'DiT-MoM-S/4':  DiT_MoM_S_4,   'DiT-MoM-S/8':  DiT_MoM_S_8,
}

# Backward compatibility
MoM = DiT_MoM
MoM_models = DiT_MoM_models
