import torch
from models import ViT_models, ViT_MoM_models

vit = ViT_models['ViT-Ti/4'](img_size=32, num_classes=10)
vit_mom = ViT_MoM_models['ViT-MoM-Ti/4'](img_size=32, num_classes=10)

# ViT: all params are active
vit_total = sum(p.numel() for p in vit.parameters())

# ViT-MoM: compute active params
mom_total = sum(p.numel() for p in vit_mom.parameters())

# Inactive = unselected experts across all blocks
block = vit_mom.blocks[0]
num_experts = block.mom.num_experts
top_k = block.mom.top_k
single_expert = sum(p.numel() for p in block.mom.experts[0].parameters())
num_blocks = len(vit_mom.blocks)

inactive_experts_per_block = (num_experts - top_k) * single_expert
total_inactive = inactive_experts_per_block * num_blocks
mom_active = mom_total - total_inactive

print(f'ViT-Ti/4:     Active = {vit_total:,} | Total = {vit_total:,}')
print(f'ViT-MoM-Ti/4: Active = {mom_active:,} | Total = {mom_total:,}')
print(f'')
print(f'Active ratio: {mom_active/vit_total:.2f}x')
print(f'Total ratio:  {mom_total/vit_total:.2f}x')
