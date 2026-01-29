# Models package for Mixture of Mixers
#
# Contains:
#   - DiT: Diffusion Transformer (baseline)
#   - MoM: Mixture of Mixers (our method)

from .dit import DiT, DiT_models
from .mom import MoM, MoM_models

# Combined model registry
all_models = {**DiT_models, **MoM_models}

__all__ = [
    'DiT', 'DiT_models',
    'MoM', 'MoM_models', 
    'all_models',
]
