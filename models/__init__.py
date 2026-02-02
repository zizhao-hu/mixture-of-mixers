# Models package for Mixture of Mixers
#
# Contains:
#   Diffusion Models:
#     - DiT: Diffusion Transformer (baseline)
#     - DiT-MoM: Diffusion Transformer with MoM replacing attention
#   
#   Classification Models:
#     - ViT: Vision Transformer (baseline)
#     - ViT-MoM: Vision Transformer with MoM replacing attention
#     - ViT-SMoM: Vision Transformer with Soft MoE token mixers
#
#   Core Components:
#     - MixtureOfMixers: Standalone attention replacement module
#     - SoftMixtureOfMixers: Soft MoE version with slot bottleneck
#     - TokenMixer: 2-layer MLP mixing across token dimension

from .dit import DiT, DiT_models
from .mom import (
    # Core module (attention replacement)
    MixtureOfMixers,
    LinearMixtureOfMixers,
    SoftMixtureOfMixers,
    TokenMixer,
    # Diffusion model
    DiT_MoM,
    DiT_MoM_models,
    # Backward compatibility
    MoM,
    MoM_models,
)
from .vit import ViT, ViT_models
from .vit_mom import ViT_MoM, ViT_MoM_models, ViT_LMoM, ViT_LMoM_models, ViT_SMoM, ViT_SMoM_models

# Combined model registries
diffusion_models = {**DiT_models, **DiT_MoM_models}
classification_models = {**ViT_models, **ViT_MoM_models, **ViT_LMoM_models, **ViT_SMoM_models}
all_models = {**diffusion_models, **classification_models}

__all__ = [
    # Diffusion baseline
    'DiT', 'DiT_models',
    # MoM components (for use in other architectures)
    'MixtureOfMixers', 'LinearMixtureOfMixers', 'SoftMixtureOfMixers', 'TokenMixer',
    # DiT with MoM
    'DiT_MoM', 'DiT_MoM_models',
    # Classification baseline
    'ViT', 'ViT_models',
    # ViT with MoM
    'ViT_MoM', 'ViT_MoM_models',
    # ViT with Linear MoM
    'ViT_LMoM', 'ViT_LMoM_models',
    # ViT with Soft MoE
    'ViT_SMoM', 'ViT_SMoM_models',
    # Backward compatibility
    'MoM', 'MoM_models',
    # Registries
    'diffusion_models', 'classification_models', 'all_models',
]
