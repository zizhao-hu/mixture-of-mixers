"""Compare ViT vs ViT-MoM parameters."""
import torch
from models import ViT_models, ViT_MoM_models

print('=' * 70)
print('Parameter Comparison: ViT vs ViT-MoM (CIFAR-10, Ti/4)')
print('=' * 70)

# Create models
vit = ViT_models['ViT-Ti/4'](img_size=32, num_classes=10)
vit_mom = ViT_MoM_models['ViT-MoM-Ti/4'](img_size=32, num_classes=10)

# Total params
vit_params = sum(p.numel() for p in vit.parameters())
mom_params = sum(p.numel() for p in vit_mom.parameters())

print(f'\nViT-Ti/4 total params: {vit_params:,}')
print(f'ViT-MoM-Ti/4 total params: {mom_params:,}')

# Get block params
print('\n' + '-' * 70)
print('Per-Block Analysis')
print('-' * 70)

# ViT block
vit_block = vit.blocks[0]
attn_params = sum(p.numel() for p in vit_block.attn.parameters())
mlp_params = sum(p.numel() for p in vit_block.mlp.parameters())
print(f'\nViT Block:')
print(f'  Attention params: {attn_params:,}')
print(f'  MLP params: {mlp_params:,}')
print(f'  Attention/MLP ratio: {attn_params/mlp_params:.4f} (target: 0.5)')

# ViT-MoM block
mom_block = vit_mom.blocks[0]
mom_module = mom_block.mom

# Total MoM params
mom_total = sum(p.numel() for p in mom_module.parameters())

# Expert params (all 64)
expert_params = sum(p.numel() for e in mom_module.experts for p in e.parameters())

# Router params
router_params = sum(p.numel() for p in mom_module.router.parameters())

# Output projection params
out_proj_params = sum(p.numel() for p in mom_module.out_proj.parameters())

# Active params (top_k=4 experts + router + out_proj)
single_expert_params = sum(p.numel() for p in mom_module.experts[0].parameters())
active_expert_params = mom_module.top_k * single_expert_params
active_mom_params = active_expert_params + router_params + out_proj_params

# MLP params in MoM block
mom_mlp_params = sum(p.numel() for p in mom_block.mlp.parameters())

print(f'\nViT-MoM Block:')
print(f'  MoM total params: {mom_total:,}')
print(f'    - All {mom_module.num_experts} experts: {expert_params:,}')
print(f'    - Single expert: {single_expert_params:,}')
print(f'    - Router: {router_params:,}')
print(f'    - Out projection: {out_proj_params:,}')
print(f'  MoM ACTIVE params (top-{mom_module.top_k}): {active_mom_params:,}')
print(f'  MLP params: {mom_mlp_params:,}')
print(f'  Active MoM/MLP ratio: {active_mom_params/mom_mlp_params:.4f} (target: 0.5)')

print(f'\n  hidden_ratio used: {mom_module.hidden_ratio:.4f}')
print(f'  D={mom_module.hidden_size}, N={mom_module.num_tokens}')

# Breakdown
print('\n' + '-' * 70)
print('Breakdown vs Target')
print('-' * 70)
print(f'  Active expert params: {active_expert_params:,}')
print(f'  Target (3/8 * MLP): {int(3/8 * mom_mlp_params):,}')
print(f'  Out proj params: {out_proj_params:,}')
print(f'  Target (1/8 * MLP): {int(1/8 * mom_mlp_params):,}')
