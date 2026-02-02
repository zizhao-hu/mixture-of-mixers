# Vision Transformer with Mixture of Mixers (ViT-MoM) for Classification
#
# ViT with attention replaced by MoE-style token mixers.
# Token mixers: 2-layer MLP mixing across spatial dimension (N)

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import trunc_normal_

from .mom import MixtureOfMixers, LinearMixtureOfMixers
from .vit import get_2d_sincos_pos_embed


class ViT_MoMBlock(nn.Module):
    """
    ViT block with MoM replacing attention.
    
    Uses pre-norm architecture like standard ViT.
    """
    def __init__(
        self,
        dim,
        num_tokens,
        num_heads=8,
        num_experts=8,
        top_k=2,
        mlp_ratio=4.,
        drop=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mom = MixtureOfMixers(
            hidden_size=dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # MoM with pre-norm
        normed = self.norm1(x)
        mom_out, aux_loss = self.mom(normed)
        x = x + mom_out
        
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x, aux_loss


class ViT_MoM(nn.Module):
    """
    Vision Transformer with Mixture of Mixers (ViT-MoM).
    
    This is ViT with attention replaced by MoE-style token mixers.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of classification classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_experts: Number of token mixer experts
        top_k: Number of experts to select
        mlp_ratio: MLP hidden dimension ratio (also used to auto-balance MoM)
        drop_rate: Dropout rate
        aux_loss_weight: Weight for load balancing auxiliary loss
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
        num_experts=8,
        top_k=2,
        mlp_ratio=4.,
        drop_rate=0.,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.aux_loss_weight = aux_loss_weight

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        num_tokens = num_patches + 1  # +1 for cls token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            ViT_MoMBlock(
                dim=embed_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
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

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        return x[:, 0], total_aux_loss / self.depth  # Return cls token and avg aux loss

    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            logits: Classification logits (B, num_classes)
            aux_loss: Load balancing auxiliary loss (scalar)
        """
        features, aux_loss = self.forward_features(x)
        logits = self.head(features)
        return logits, aux_loss


#################################################################################
#                            ViT-MoM Configurations                              #
#################################################################################

def ViT_MoM_Ti_16(**kwargs):
    """ViT-MoM-Tiny with patch size 16."""
    return ViT_MoM(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)

def ViT_MoM_S_16(**kwargs):
    """ViT-MoM-Small with patch size 16."""
    return ViT_MoM(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)

def ViT_MoM_B_16(**kwargs):
    """ViT-MoM-Base with patch size 16."""
    return ViT_MoM(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

def ViT_MoM_L_16(**kwargs):
    """ViT-MoM-Large with patch size 16."""
    return ViT_MoM(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)

# Smaller patch sizes (more tokens)
def ViT_MoM_Ti_8(**kwargs):
    """ViT-MoM-Tiny with patch size 8."""
    return ViT_MoM(patch_size=8, embed_dim=192, depth=12, num_heads=3, **kwargs)

def ViT_MoM_S_8(**kwargs):
    """ViT-MoM-Small with patch size 8."""
    return ViT_MoM(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)

def ViT_MoM_B_8(**kwargs):
    """ViT-MoM-Base with patch size 8."""
    return ViT_MoM(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)

# For CIFAR (32x32 images, use patch size 4)
def ViT_MoM_Ti_4(**kwargs):
    """ViT-MoM-Tiny with patch size 4 (for CIFAR)."""
    return ViT_MoM(patch_size=4, embed_dim=192, depth=12, num_heads=3, num_experts=32, top_k=4, **kwargs)

def ViT_MoM_S_4(**kwargs):
    """ViT-MoM-Small with patch size 4 (for CIFAR)."""
    return ViT_MoM(patch_size=4, embed_dim=384, depth=12, num_heads=6, num_experts=32, top_k=4, **kwargs)

def ViT_MoM_B_4(**kwargs):
    """ViT-MoM-Base with patch size 4 (for CIFAR)."""
    return ViT_MoM(patch_size=4, embed_dim=768, depth=12, num_heads=12, num_experts=32, top_k=4, **kwargs)


ViT_MoM_models = {
    # Standard patch size 16 (for 224x224)
    'ViT-MoM-Ti/16': ViT_MoM_Ti_16,
    'ViT-MoM-S/16': ViT_MoM_S_16,
    'ViT-MoM-B/16': ViT_MoM_B_16,
    'ViT-MoM-L/16': ViT_MoM_L_16,
    # Patch size 8 (for 224x224, more tokens)
    'ViT-MoM-Ti/8': ViT_MoM_Ti_8,
    'ViT-MoM-S/8': ViT_MoM_S_8,
    'ViT-MoM-B/8': ViT_MoM_B_8,
    # Patch size 4 (for CIFAR 32x32)
    'ViT-MoM-Ti/4': ViT_MoM_Ti_4,
    'ViT-MoM-S/4': ViT_MoM_S_4,
    'ViT-MoM-B/4': ViT_MoM_B_4,
}


#################################################################################
#                    ViT with Soft MoE Token Mixers (ViT-SMoM)                   #
#################################################################################

from .mom import SoftMixtureOfMixers

class ViT_SMoMBlock(nn.Module):
    """
    ViT block with Soft MoE token mixers (slot bottleneck).
    """
    def __init__(
        self,
        dim,
        num_tokens,
        num_experts=8,
        slots_per_expert=1,
        mlp_ratio=4.,
        drop=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.smom = SoftMixtureOfMixers(
            hidden_size=dim,
            num_tokens=num_tokens,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        normed = self.norm1(x)
        smom_out, aux_loss = self.smom(normed)
        x = x + smom_out
        x = x + self.mlp(self.norm2(x))
        return x, aux_loss


class ViT_SMoM(nn.Module):
    """
    Vision Transformer with Soft Mixture of Mixers (ViT-SMoM).
    
    Uses Soft MoE with slot bottleneck for compute efficiency at scale.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_experts=8,
        slots_per_expert=1,
        mlp_ratio=4.,
        drop_rate=0.,
        aux_loss_weight=0.0,  # No aux loss for soft MoE
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.aux_loss_weight = aux_loss_weight

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        num_tokens = num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            ViT_SMoMBlock(
                dim=embed_dim,
                num_tokens=num_tokens,
                num_experts=num_experts,
                slots_per_expert=slots_per_expert,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_aux_loss=True):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        x = self.head(x[:, 0])

        if return_aux_loss:
            return x, self.aux_loss_weight * total_aux_loss
        return x


# ViT-SMoM model variants
def ViT_SMoM_Ti_4(**kwargs):
    """ViT-SMoM-Tiny with patch size 4 (for CIFAR)."""
    return ViT_SMoM(patch_size=4, embed_dim=192, depth=12, **kwargs)

def ViT_SMoM_S_4(**kwargs):
    """ViT-SMoM-Small with patch size 4 (for CIFAR)."""
    return ViT_SMoM(patch_size=4, embed_dim=384, depth=12, **kwargs)

def ViT_SMoM_Ti_16(**kwargs):
    """ViT-SMoM-Tiny with patch size 16."""
    return ViT_SMoM(patch_size=16, embed_dim=192, depth=12, **kwargs)


ViT_SMoM_models = {
    'ViT-SMoM-Ti/4': ViT_SMoM_Ti_4,
    'ViT-SMoM-S/4': ViT_SMoM_S_4,
    'ViT-SMoM-Ti/16': ViT_SMoM_Ti_16,
}


class ViT_LMoMBlock(nn.Module):
    """
    ViT block with Linear MoM (N x N learned attention) replacing attention.
    """
    def __init__(
        self,
        dim,
        num_tokens,
        num_heads=8,
        num_experts=8,
        top_k=2,
        mlp_ratio=4.,
        drop=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.lmom = LinearMixtureOfMixers(
            hidden_size=dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        normed = self.norm1(x)
        lmom_out, aux_loss = self.lmom(normed)
        x = x + lmom_out
        x = x + self.mlp(self.norm2(x))
        return x, aux_loss


class ViT_LMoM(nn.Module):
    """
    Vision Transformer with Linear Mixture of Mixers (ViT-LMoM).
    
    Uses Linear MoM (N x N learned attention maps) for spatial mixing.
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
        num_experts=8,
        top_k=2,
        mlp_ratio=4.,
        drop_rate=0.,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.aux_loss_weight = aux_loss_weight

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        num_tokens = num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            ViT_LMoMBlock(
                dim=embed_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        return x[:, 0], total_aux_loss / self.depth

    def forward(self, x):
        features, aux_loss = self.forward_features(x)
        logits = self.head(features)
        return logits, aux_loss


# ViT-LMoM model variants
def ViT_LMoM_Ti_4(**kwargs):
    """ViT-LMoM-Tiny with patch size 4 (for CIFAR)."""
    return ViT_LMoM(patch_size=4, embed_dim=192, depth=12, num_heads=3, num_experts=32, top_k=4, **kwargs)

def ViT_LMoM_S_4(**kwargs):
    """ViT-LMoM-Small with patch size 4 (for CIFAR)."""
    return ViT_LMoM(patch_size=4, embed_dim=384, depth=12, num_heads=6, num_experts=32, top_k=4, **kwargs)


ViT_LMoM_models = {
    'ViT-LMoM-Ti/4': ViT_LMoM_Ti_4,
    'ViT-LMoM-S/4': ViT_LMoM_S_4,
}
