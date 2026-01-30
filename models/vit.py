# Vision Transformer (ViT) for Classification
#
# Based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
# https://arxiv.org/abs/2010.11929

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import trunc_normal_
import math


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    """Vision Transformer block with pre-norm."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of classification classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use bias in QKV projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = 1  # cls token

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        trunc_normal_(self.cls_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # Return cls token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sin-cos positional embedding."""
    import numpy as np
    
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate sin-cos embedding from grid."""
    import numpy as np
    
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sin-cos embedding from positions."""
    import numpy as np
    
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
#                               ViT Configurations                               #
#################################################################################

def ViT_Ti_16(**kwargs):
    """ViT-Tiny with patch size 16."""
    return ViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)

def ViT_S_16(**kwargs):
    """ViT-Small with patch size 16."""
    return ViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)

def ViT_B_16(**kwargs):
    """ViT-Base with patch size 16."""
    return ViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

def ViT_L_16(**kwargs):
    """ViT-Large with patch size 16."""
    return ViT(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)

# Smaller patch sizes (more tokens)
def ViT_Ti_8(**kwargs):
    """ViT-Tiny with patch size 8."""
    return ViT(patch_size=8, embed_dim=192, depth=12, num_heads=3, **kwargs)

def ViT_S_8(**kwargs):
    """ViT-Small with patch size 8."""
    return ViT(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)

def ViT_B_8(**kwargs):
    """ViT-Base with patch size 8."""
    return ViT(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)

# For CIFAR (32x32 images, use patch size 4)
def ViT_Ti_4(**kwargs):
    """ViT-Tiny with patch size 4 (for CIFAR)."""
    return ViT(patch_size=4, embed_dim=192, depth=12, num_heads=3, **kwargs)

def ViT_S_4(**kwargs):
    """ViT-Small with patch size 4 (for CIFAR)."""
    return ViT(patch_size=4, embed_dim=384, depth=12, num_heads=6, **kwargs)

def ViT_B_4(**kwargs):
    """ViT-Base with patch size 4 (for CIFAR)."""
    return ViT(patch_size=4, embed_dim=768, depth=12, num_heads=12, **kwargs)


ViT_models = {
    # Standard patch size 16 (for 224x224)
    'ViT-Ti/16': ViT_Ti_16,
    'ViT-S/16': ViT_S_16,
    'ViT-B/16': ViT_B_16,
    'ViT-L/16': ViT_L_16,
    # Patch size 8 (for 224x224, more tokens)
    'ViT-Ti/8': ViT_Ti_8,
    'ViT-S/8': ViT_S_8,
    'ViT-B/8': ViT_B_8,
    # Patch size 4 (for CIFAR 32x32)
    'ViT-Ti/4': ViT_Ti_4,
    'ViT-S/4': ViT_S_4,
    'ViT-B/4': ViT_B_4,
}
