# Copyright (c) 2024
# Mixture of Mixers (MoM) - A DiT variant replacing attention with MoE-style mixers
#
# This architecture replaces the attention mechanism in DiT with a Mixture of Experts
# approach using two types of mixers:
#   - Token Mixers: Mix information across the token/spatial dimension (N)
#   - Channel Mixers: Mix information across the channel/feature dimension (D)
#
# References:
#   - DiT: https://github.com/facebookresearch/DiT
#   - MoE: https://arxiv.org/abs/1701.06538
#   - MLP-Mixer: https://arxiv.org/abs/2105.01601

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                              Mixer Expert Modules                              #
#################################################################################

class TokenMixer(nn.Module):
    """
    Token Mixer: A 2-layer FFN that mixes across the token/spatial dimension.
    
    Input shape: (B, N, D)
    Operation: Transpose to (B, D, N) -> FFN(N -> hidden -> N) -> Transpose back to (B, N, D)
    
    This allows information to flow between different spatial positions.
    """
    def __init__(self, num_tokens, hidden_ratio=4.0):
        super().__init__()
        hidden_dim = int(num_tokens * hidden_ratio)
        self.fc1 = nn.Linear(num_tokens, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, num_tokens)
    
    def forward(self, x):
        # x: (B, N, D)
        x = x.transpose(1, 2)  # (B, D, N)
        x = self.fc1(x)        # (B, D, hidden)
        x = self.act(x)
        x = self.fc2(x)        # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class ChannelMixer(nn.Module):
    """
    Channel Mixer: A 2-layer FFN that mixes across the channel/feature dimension.
    
    Input shape: (B, N, D)
    Operation: FFN(D -> hidden -> D)
    
    This allows information to flow between different feature channels.
    """
    def __init__(self, hidden_size, hidden_ratio=4.0):
        super().__init__()
        hidden_dim = int(hidden_size * hidden_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, hidden_size)
    
    def forward(self, x):
        # x: (B, N, D)
        x = self.fc1(x)   # (B, N, hidden)
        x = self.act(x)
        x = self.fc2(x)   # (B, N, D)
        return x


#################################################################################
#                           Mixture of Mixers (MoE Layer)                        #
#################################################################################

class MixtureOfMixers(nn.Module):
    """
    Mixture of Mixers: Replaces attention with MoE-style routing over mixer experts.
    
    Architecture:
        - num_token_experts Token Mixers (default: 4)
        - num_channel_experts Channel Mixers (default: 4)
        - Router that selects top-k experts (default: k=2)
        - Weighted sum of selected expert outputs
    
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_token_experts=4,
        num_channel_experts=4,
        top_k=2,
        token_hidden_ratio=4.0,
        channel_hidden_ratio=4.0,
        router_jitter_noise=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_token_experts = num_token_experts
        self.num_channel_experts = num_channel_experts
        self.num_experts = num_token_experts + num_channel_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        
        # Create Token Mixer experts
        self.token_mixers = nn.ModuleList([
            TokenMixer(num_tokens, hidden_ratio=token_hidden_ratio)
            for _ in range(num_token_experts)
        ])
        
        # Create Channel Mixer experts
        self.channel_mixers = nn.ModuleList([
            ChannelMixer(hidden_size, hidden_ratio=channel_hidden_ratio)
            for _ in range(num_channel_experts)
        ])
        
        # Router: projects input to expert scores
        # We use mean pooling over tokens for routing decision
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, D)
        
        Returns:
            output: Tensor of shape (B, N, D)
            aux_loss: Load balancing auxiliary loss for training
        """
        B, N, D = x.shape
        
        # Compute routing scores using mean-pooled representation
        # (B, N, D) -> (B, D) -> (B, num_experts)
        router_input = x.mean(dim=1)  # Global average pooling
        
        # Add jitter noise during training for exploration
        if self.training and self.router_jitter_noise > 0:
            router_input = router_input * (1 + torch.randn_like(router_input) * self.router_jitter_noise)
        
        router_logits = self.router(router_input)  # (B, num_experts)
        
        # Get top-k experts and their weights
        router_probs = F.softmax(router_logits, dim=-1)  # (B, num_experts)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (B, k)
        
        # Normalize weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # (B, k)
        
        # Compute expert outputs and weighted sum
        output = torch.zeros_like(x)  # (B, N, D)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]  # (B,)
            weights = top_k_weights[:, i]  # (B,)
            
            # Process each sample through its selected expert
            for b in range(B):
                expert_idx = expert_indices[b].item()
                weight = weights[b]
                
                if expert_idx < self.num_token_experts:
                    # Token mixer
                    expert_output = self.token_mixers[expert_idx](x[b:b+1])
                else:
                    # Channel mixer
                    channel_idx = expert_idx - self.num_token_experts
                    expert_output = self.channel_mixers[channel_idx](x[b:b+1])
                
                output[b:b+1] += weight * expert_output
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _compute_aux_loss(self, router_probs, top_k_indices):
        """
        Compute load balancing auxiliary loss to encourage even expert utilization.
        
        Based on Switch Transformer: https://arxiv.org/abs/2101.03961
        """
        # Fraction of tokens routed to each expert
        num_experts = self.num_experts
        
        # One-hot encode the top-1 expert selection
        top1_indices = top_k_indices[:, 0]  # (B,)
        expert_mask = F.one_hot(top1_indices, num_classes=num_experts).float()  # (B, num_experts)
        
        # Average probability assigned to each expert
        expert_probs = router_probs.mean(dim=0)  # (num_experts,)
        
        # Fraction of samples assigned to each expert
        expert_fraction = expert_mask.mean(dim=0)  # (num_experts,)
        
        # Load balancing loss: encourages equal routing
        aux_loss = num_experts * (expert_probs * expert_fraction).sum()
        
        return aux_loss


class MixtureOfMixersFast(nn.Module):
    """
    Optimized Mixture of Mixers with batched expert computation.
    
    This version groups samples by their selected experts for more efficient
    batched computation on GPU.
    """
    def __init__(
        self,
        hidden_size,
        num_tokens,
        num_token_experts=4,
        num_channel_experts=4,
        top_k=2,
        token_hidden_ratio=4.0,
        channel_hidden_ratio=4.0,
        router_jitter_noise=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_token_experts = num_token_experts
        self.num_channel_experts = num_channel_experts
        self.num_experts = num_token_experts + num_channel_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        
        # Create Token Mixer experts
        self.token_mixers = nn.ModuleList([
            TokenMixer(num_tokens, hidden_ratio=token_hidden_ratio)
            for _ in range(num_token_experts)
        ])
        
        # Create Channel Mixer experts
        self.channel_mixers = nn.ModuleList([
            ChannelMixer(hidden_size, hidden_ratio=channel_hidden_ratio)
            for _ in range(num_channel_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        
    def forward(self, x):
        B, N, D = x.shape
        
        # Compute routing scores
        router_input = x.mean(dim=1)
        if self.training and self.router_jitter_noise > 0:
            router_input = router_input * (1 + torch.randn_like(router_input) * self.router_jitter_noise)
        
        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert (batched over samples that selected it)
        for expert_idx in range(self.num_experts):
            # Find which (batch, top_k_position) pairs selected this expert
            mask = (top_k_indices == expert_idx)  # (B, top_k)
            
            if not mask.any():
                continue
            
            # Get the weights for samples that selected this expert
            weights = (top_k_weights * mask.float()).sum(dim=-1)  # (B,)
            active_samples = weights > 0
            
            if not active_samples.any():
                continue
            
            # Get active inputs
            active_x = x[active_samples]  # (num_active, N, D)
            active_weights = weights[active_samples].view(-1, 1, 1)  # (num_active, 1, 1)
            
            # Compute expert output
            if expert_idx < self.num_token_experts:
                expert_output = self.token_mixers[expert_idx](active_x)
            else:
                channel_idx = expert_idx - self.num_token_experts
                expert_output = self.channel_mixers[channel_idx](active_x)
            
            # Add weighted contribution
            output[active_samples] += active_weights * expert_output
        
        # Compute auxiliary loss
        aux_loss = self._compute_aux_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _compute_aux_loss(self, router_probs, top_k_indices):
        num_experts = self.num_experts
        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, num_classes=num_experts).float()
        expert_probs = router_probs.mean(dim=0)
        expert_fraction = expert_mask.mean(dim=0)
        aux_loss = num_experts * (expert_probs * expert_fraction).sum()
        return aux_loss


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with dropout for CFG."""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                              MoM Block (replaces DiTBlock)                     #
#################################################################################

class MoMBlock(nn.Module):
    """
    Mixture of Mixers Block with adaptive layer norm zero (adaLN-Zero) conditioning.
    
    Replaces attention in DiTBlock with MixtureOfMixers.
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
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Replace attention with Mixture of Mixers
        self.mixer = MixtureOfMixersFast(
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            num_token_experts=num_token_experts,
            num_channel_experts=num_channel_experts,
            top_k=top_k,
            token_hidden_ratio=mixer_hidden_ratio,
            channel_hidden_ratio=mixer_hidden_ratio,
        )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # adaLN modulation (same as DiT)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor (B, N, D)
            c: Conditioning tensor (B, D)
        
        Returns:
            x: Output tensor (B, N, D)
            aux_loss: Auxiliary loss from MoE routing
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Mixer branch (replaces attention)
        mixer_out, aux_loss = self.mixer(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa.unsqueeze(1) * mixer_out
        
        # MLP branch (same as DiT)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x, aux_loss


class FinalLayer(nn.Module):
    """The final layer of MoM (same as DiT)."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                            Mixture of Mixers Model                             #
#################################################################################

class MoM(nn.Module):
    """
    Mixture of Mixers (MoM): A DiT variant replacing attention with MoE-style mixers.
    
    Key differences from DiT:
        - Attention is replaced with MixtureOfMixers
        - Returns auxiliary loss for load balancing during training
        - Similar parameter count but different computational pattern
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

        # Initialize pos_embed with sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize embeddings
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
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
            output: (B, out_channels, H, W) predicted noise/variance
            aux_loss: Auxiliary load balancing loss (only meaningful during training)
        """
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)
        t = self.t_embedder(t)                   # (B, D)
        y = self.y_embedder(y, self.training)    # (B, D)
        c = t + y                                # (B, D)
        
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, c)
            total_aux_loss = total_aux_loss + aux_loss
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        # Average auxiliary loss over all blocks
        total_aux_loss = total_aux_loss / self.depth
        
        return x, total_aux_loss

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
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


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


#################################################################################
#                                  Testing                                       #
#################################################################################

if __name__ == "__main__":
    # Test MoM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing MoM on {device}")
    
    # Create model
    model = MoM_S_4(input_size=32, num_classes=1000).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"MoM-S/4 Parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 4, 32, 32).to(device)  # Latent input
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # Timesteps
    y = torch.randint(0, 1000, (batch_size,)).to(device)  # Class labels
    
    model.train()
    output, aux_loss = model(x, t, y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    # Test inference (no aux loss needed)
    model.eval()
    with torch.no_grad():
        output, _ = model(x, t, y)
    print(f"Inference output shape: {output.shape}")
    
    print("\nMoM model test passed!")
