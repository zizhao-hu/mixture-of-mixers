# Mixture of Mixers (MoM)
# A DiT variant replacing attention with MoE-style token/channel mixers
#
# Architecture:
#   - Token Mixers: Mix information across the token/spatial dimension (N)
#   - Channel Mixers: Mix information across the channel/feature dimension (D)
#   - Router selects top-k experts from the pool
#   - Weighted sum of selected expert outputs
#
# Key design: Each mixer type has its own normalization direction
#   - Token Mixer: TokenNorm (normalizes across N dimension)
#   - Channel Mixer: LayerNorm (normalizes across D dimension)

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
#                              Normalization Layers                              #
#################################################################################

class TokenNorm(nn.Module):
    """
    Token-wise normalization: normalizes across the token/spatial dimension (N).
    
    For input (B, N, D), normalizes over dimension 1 (tokens).
    Each channel is normalized independently across all spatial positions.
    """
    def __init__(self, num_tokens, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_tokens))
            self.bias = nn.Parameter(torch.zeros(num_tokens))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # x: (B, N, D)
        # Normalize over N dimension for each channel independently
        x = x.transpose(1, 2)  # (B, D, N)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class ChannelNorm(nn.Module):
    """
    Channel-wise normalization: normalizes across the channel/feature dimension (D).
    
    This is equivalent to standard LayerNorm over the last dimension.
    For input (B, N, D), normalizes over dimension 2 (channels).
    """
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        # x: (B, N, D)
        return self.norm(x)


#################################################################################
#                              Mixer Expert Modules                              #
#################################################################################

class TokenMixer(nn.Module):
    """
    Token Mixer: A 2-layer FFN that mixes across the token/spatial dimension.
    
    Input shape: (B, N, D)
    Operation: 
        1. TokenNorm (normalize across N)
        2. Transpose to (B, D, N)
        3. FFN(N -> hidden -> N)
        4. Transpose back to (B, N, D)
    """
    def __init__(self, num_tokens, hidden_size, hidden_ratio=4.0):
        super().__init__()
        self.norm = TokenNorm(num_tokens, elementwise_affine=False)
        hidden_dim = int(num_tokens * hidden_ratio)
        self.fc1 = nn.Linear(num_tokens, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, num_tokens)
    
    def forward(self, x):
        # x: (B, N, D)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, D, N)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class ChannelMixer(nn.Module):
    """
    Channel Mixer: A 2-layer FFN that mixes across the channel/feature dimension.
    
    Input shape: (B, N, D)
    Operation:
        1. ChannelNorm (normalize across D)
        2. FFN(D -> hidden -> D)
    """
    def __init__(self, num_tokens, hidden_size, hidden_ratio=4.0):
        super().__init__()
        self.norm = ChannelNorm(hidden_size, elementwise_affine=False)
        hidden_dim = int(hidden_size * hidden_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, hidden_size)
    
    def forward(self, x):
        # x: (B, N, D)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                           Mixture of Mixers (MoE Layer)                        #
#################################################################################

class MixtureOfMixers(nn.Module):
    """
    Mixture of Mixers: MoE-style routing over token and channel mixer experts.
    
    Each expert has its own normalization appropriate for its mixing direction:
        - Token Mixers use TokenNorm (normalizes across spatial dimension)
        - Channel Mixers use ChannelNorm (normalizes across feature dimension)
    
    Args:
        hidden_size: Feature dimension (D)
        num_tokens: Number of tokens (N)
        num_token_experts: Number of token mixer experts
        num_channel_experts: Number of channel mixer experts
        top_k: Number of experts to select per input
        mixer_hidden_ratio: Hidden layer ratio for mixers
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_token_experts=4,
        num_channel_experts=4,
        top_k=2,
        mixer_hidden_ratio=4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_token_experts = num_token_experts
        self.num_channel_experts = num_channel_experts
        self.num_experts = num_token_experts + num_channel_experts
        self.top_k = top_k
        
        # Create experts (each with its own appropriate normalization)
        self.token_mixers = nn.ModuleList([
            TokenMixer(num_tokens, hidden_size, hidden_ratio=mixer_hidden_ratio)
            for _ in range(num_token_experts)
        ])
        self.channel_mixers = nn.ModuleList([
            ChannelMixer(num_tokens, hidden_size, hidden_ratio=mixer_hidden_ratio)
            for _ in range(num_channel_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, N, D)
        
        Returns:
            output: Tensor (B, N, D)
            aux_loss: Load balancing loss
        """
        B, N, D = x.shape
        
        # Compute routing scores (global average pooling for routing decision)
        router_input = x.mean(dim=1)  # (B, D)
        router_logits = self.router(router_input)  # (B, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            # Find samples that selected this expert
            mask = (top_k_indices == expert_idx)
            weights = (top_k_weights * mask.float()).sum(dim=-1)
            active = weights > 0
            
            if not active.any():
                continue
            
            # Compute expert output for active samples
            active_x = x[active]
            active_weights = weights[active].view(-1, 1, 1)
            
            if expert_idx < self.num_token_experts:
                expert_out = self.token_mixers[expert_idx](active_x)
            else:
                expert_out = self.channel_mixers[expert_idx - self.num_token_experts](active_x)
            
            output[active] += active_weights * expert_out
        
        # Load balancing auxiliary loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _load_balancing_loss(self, router_probs, top_k_indices):
        """Compute load balancing loss to encourage even expert usage."""
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=self.num_experts).float()
        expert_probs = router_probs.mean(dim=0)
        expert_fraction = expert_mask.mean(dim=0)
        return self.num_experts * (expert_probs * expert_fraction).sum()


#################################################################################
#                              MoM Block                                         #
#################################################################################

class MoMBlock(nn.Module):
    """
    MoM Block with adaptive layer norm zero (adaLN-Zero) conditioning.
    
    Unlike DiT which uses a single LayerNorm before attention,
    MoM delegates normalization to each expert (TokenNorm or ChannelNorm).
    
    The block still uses LayerNorm before the MLP (standard practice).
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_token_experts=4,
        num_channel_experts=4,
        top_k=2,
        mlp_ratio=4.0,
        mixer_hidden_ratio=4.0,
    ):
        super().__init__()
        # No norm1 here - each expert handles its own normalization
        self.mixer = MixtureOfMixers(
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            num_token_experts=num_token_experts,
            num_channel_experts=num_channel_experts,
            top_k=top_k,
            mixer_hidden_ratio=mixer_hidden_ratio,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # adaLN modulation - 4 params for mixer (shift, scale, gate) + 3 for mlp
        # Note: we still modulate the input to mixer, but norm is inside each expert
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Modulate input before passing to mixer (mixer handles its own norm)
        modulated_x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        mixer_out, aux_loss = self.mixer(modulated_x)
        x = x + gate_msa.unsqueeze(1) * mixer_out
        
        # Standard MLP path with LayerNorm
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x, aux_loss


#################################################################################
#                            Mixture of Mixers Model                             #
#################################################################################

class MoM(nn.Module):
    """
    Mixture of Mixers (MoM): DiT with attention replaced by MoE-style mixers.
    
    Key differences from DiT:
        - Attention replaced with MixtureOfMixers
        - Each mixer expert has its own normalization:
            - Token Mixers: TokenNorm (normalizes across spatial dim)
            - Channel Mixers: ChannelNorm (normalizes across feature dim)
        - Returns auxiliary loss for load balancing during training
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_token_experts=4,
        num_channel_experts=4,
        top_k=2,
        mlp_ratio=4.0,
        mixer_hidden_ratio=4.0,
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
            MoMBlock(
                hidden_size=hidden_size,
                num_tokens=num_patches,
                num_token_experts=num_token_experts,
                num_channel_experts=num_channel_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                mixer_hidden_ratio=mixer_hidden_ratio,
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
        """
        Forward pass of MoM.
        
        Args:
            x: (B, C, H, W) tensor of spatial inputs
            t: (B,) tensor of diffusion timesteps
            y: (B,) tensor of class labels
        
        Returns:
            output: (B, out_channels, H, W)
            aux_loss: Load balancing auxiliary loss
        """
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
        """Forward pass with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, _ = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                 MoM Configs                                    #
#################################################################################

def MoM_XL_2(**kwargs):
    return MoM(depth=28, hidden_size=1152, patch_size=2, **kwargs)

def MoM_XL_4(**kwargs):
    return MoM(depth=28, hidden_size=1152, patch_size=4, **kwargs)

def MoM_XL_8(**kwargs):
    return MoM(depth=28, hidden_size=1152, patch_size=8, **kwargs)

def MoM_L_2(**kwargs):
    return MoM(depth=24, hidden_size=1024, patch_size=2, **kwargs)

def MoM_L_4(**kwargs):
    return MoM(depth=24, hidden_size=1024, patch_size=4, **kwargs)

def MoM_L_8(**kwargs):
    return MoM(depth=24, hidden_size=1024, patch_size=8, **kwargs)

def MoM_B_2(**kwargs):
    return MoM(depth=12, hidden_size=768, patch_size=2, **kwargs)

def MoM_B_4(**kwargs):
    return MoM(depth=12, hidden_size=768, patch_size=4, **kwargs)

def MoM_B_8(**kwargs):
    return MoM(depth=12, hidden_size=768, patch_size=8, **kwargs)

def MoM_S_2(**kwargs):
    return MoM(depth=12, hidden_size=384, patch_size=2, **kwargs)

def MoM_S_4(**kwargs):
    return MoM(depth=12, hidden_size=384, patch_size=4, **kwargs)

def MoM_S_8(**kwargs):
    return MoM(depth=12, hidden_size=384, patch_size=8, **kwargs)


MoM_models = {
    'MoM-XL/2': MoM_XL_2,  'MoM-XL/4': MoM_XL_4,  'MoM-XL/8': MoM_XL_8,
    'MoM-L/2':  MoM_L_2,   'MoM-L/4':  MoM_L_4,   'MoM-L/8':  MoM_L_8,
    'MoM-B/2':  MoM_B_2,   'MoM-B/4':  MoM_B_4,   'MoM-B/8':  MoM_B_8,
    'MoM-S/2':  MoM_S_2,   'MoM-S/4':  MoM_S_4,   'MoM-S/8':  MoM_S_8,
}
