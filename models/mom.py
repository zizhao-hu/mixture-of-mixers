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


#################################################################################
#                    Mixture of Mixers (Attention Replacement)                   #
#################################################################################

class MixtureOfMixers(nn.Module):
    """
    Mixture of Mixers: Drop-in attention replacement using MoE token mixers.
    
    Auto-computes num_experts and top_k to match attention parameters:
        - top_k selected so active params ≈ attention params (4D²)
        - num_experts = 10 × top_k (sparse MoE)
    
    Args:
        hidden_size: Feature dimension (D)
        num_tokens: Number of tokens (N)
        hidden_ratio: Hidden ratio for token mixer MLPs
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        hidden_ratio=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.hidden_ratio = hidden_ratio
        
        # Auto-compute top_k to match attention params (4D²)
        # Active params: top_k experts + out_proj + router
        #   - Experts: top_k × 2 × hidden_ratio × N²
        #   - Out proj: D²
        #   - Router: num_experts × D = 10 × top_k × D
        # Target: 4D² (attention = Wq, Wk, Wv, Wo)
        # top_k × (2 × hidden_ratio × N² + 10D) + D² ≤ 4D²
        # top_k ≤ 3D² / (2 × hidden_ratio × N² + 10D)
        D, N = hidden_size, num_tokens
        top_k = int(3 * D**2 / (2 * hidden_ratio * N**2 + 10 * D))  # floor
        top_k = max(1, top_k)  # At least 1 expert
        
        # Total experts = 10 × selected experts
        num_experts = 10 * top_k
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Token mixer experts
        self.experts = nn.ModuleList([
            TokenMixer(num_tokens, hidden_size, hidden_ratio=hidden_ratio)
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
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
        
        # Compute routing scores
        router_input = x.mean(dim=1)  # (B, D)
        router_logits = self.router(router_input)  # (B, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx)
            weights = (top_k_weights * mask.float()).sum(dim=-1)
            active = weights > 0
            
            if not active.any():
                continue
            
            active_x = x[active]
            active_weights = weights[active].view(-1, 1, 1)
            expert_out = self.experts[expert_idx](active_x)
            output[active] += active_weights * expert_out
        
        # Output projection
        output = self.out_proj(output)
        
        # Load balancing loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _load_balancing_loss(self, router_probs, top_k_indices):
        """Load balancing loss to encourage even expert usage."""
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=self.num_experts).float()
        expert_probs = router_probs.mean(dim=0)
        expert_fraction = expert_mask.mean(dim=0)
        return self.num_experts * (expert_probs * expert_fraction).sum()


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
    ):
        super().__init__()
        self.mom = MixtureOfMixers(
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            hidden_ratio=hidden_ratio,
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
