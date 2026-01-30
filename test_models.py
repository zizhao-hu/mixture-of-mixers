"""Test and compare DiT and MoM models."""
import torch
from models import DiT_models, MoM_models

def main():
    print('='*70)
    print('MODEL COMPARISON: DiT vs MoM')
    print('='*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print()

    # Test configurations
    configs = [
        ('S/4', 32, 384, 12),   # Small
        ('B/4', 32, 768, 12),   # Base
    ]

    for name, input_size, hidden_size, depth in configs:
        print(f'--- {name} Configuration (input_size={input_size}, hidden={hidden_size}, depth={depth}) ---')
        print()
        
        # Create models
        dit = DiT_models[f'DiT-{name}'](input_size=input_size, num_classes=1000).to(device)
        mom = MoM_models[f'MoM-{name}'](input_size=input_size, num_classes=1000).to(device)
        
        # Count parameters
        dit_params = sum(p.numel() for p in dit.parameters())
        mom_params = sum(p.numel() for p in mom.parameters())
        
        dit_trainable = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        mom_trainable = sum(p.numel() for p in mom.parameters() if p.requires_grad)
        
        print(f'DiT-{name}:')
        print(f'  Total params:     {dit_params:>12,}')
        print(f'  Trainable params: {dit_trainable:>12,}')
        print()
        
        print(f'MoM-{name}:')
        print(f'  Total params:     {mom_params:>12,}')
        print(f'  Trainable params: {mom_trainable:>12,}')
        print(f'  Ratio vs DiT:     {mom_params/dit_params:.2f}x')
        print()
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 4, input_size, input_size).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        y = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # DiT forward
        dit.train()
        dit_out = dit(x, t, y)
        print(f'DiT-{name} Forward:')
        print(f'  Input:  {tuple(x.shape)}')
        print(f'  Output: {tuple(dit_out.shape)}')
        
        # MoM forward
        mom.train()
        mom_out, aux_loss = mom(x, t, y)
        print(f'MoM-{name} Forward:')
        print(f'  Input:  {tuple(x.shape)}')
        print(f'  Output: {tuple(mom_out.shape)}')
        print(f'  Aux Loss: {aux_loss.item():.4f}')
        print()
        
        # Memory usage (approximate)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            _ = dit(x, t, y)
            dit_mem = torch.cuda.max_memory_allocated() / 1024**2
            
            torch.cuda.reset_peak_memory_stats()
            _ = mom(x, t, y)
            mom_mem = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f'Peak GPU Memory (batch={batch_size}):')
            print(f'  DiT: {dit_mem:.1f} MB')
            print(f'  MoM: {mom_mem:.1f} MB')
            print()
        
        # Clean up
        del dit, mom
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print('='*70)
        print()

    # Detailed breakdown for MoM-S/4
    print('DETAILED BREAKDOWN: MoM-S/4')
    print('='*70)
    mom = MoM_models['MoM-S/4'](input_size=32, num_classes=1000)

    print('\nParameter breakdown by component:')
    components = {
        'x_embedder': 0,
        't_embedder': 0,
        'y_embedder': 0,
        'blocks': 0,
        'final_layer': 0,
        'pos_embed': 0,
    }

    for name, param in mom.named_parameters():
        for comp in components:
            if comp in name:
                components[comp] += param.numel()
                break

    for comp, count in components.items():
        print(f'  {comp:15s}: {count:>10,} ({100*count/sum(components.values()):.1f}%)')

    print(f'  {"Total":15s}: {sum(components.values()):>10,}')

    # Block breakdown
    print('\nPer-block breakdown (MoMBlock):')
    block = mom.blocks[0]
    block_components = {}
    for name, param in block.named_parameters():
        comp = name.split('.')[0]
        if comp not in block_components:
            block_components[comp] = 0
        block_components[comp] += param.numel()

    for comp, count in block_components.items():
        print(f'  {comp:20s}: {count:>10,}')

    # Expert breakdown
    print('\nMixer configuration:')
    print(f'  Token mixers:   {mom.blocks[0].mixer.num_token_experts}')
    print(f'  Channel mixers: {mom.blocks[0].mixer.num_channel_experts}')
    print(f'  Top-k:          {mom.blocks[0].mixer.top_k}')
    print(f'  Total experts:  {mom.blocks[0].mixer.num_experts}')

    # Router analysis
    print('\nRouter test (which experts get selected):')
    mom.eval()
    x = torch.randn(8, 4, 32, 32)
    t = torch.randint(0, 1000, (8,))
    y = torch.randint(0, 1000, (8,))
    
    # Hook to capture router outputs
    router_outputs = []
    def hook(module, input, output):
        router_outputs.append(output)
    
    mom.blocks[0].mixer.router.register_forward_hook(hook)
    with torch.no_grad():
        _ = mom(x, t, y)
    
    router_logits = router_outputs[0]
    router_probs = torch.softmax(router_logits, dim=-1)
    top_k_weights, top_k_indices = torch.topk(router_probs, 2, dim=-1)
    
    print(f'  Sample routing (batch of 8):')
    for i in range(min(4, len(top_k_indices))):
        experts = top_k_indices[i].tolist()
        weights = top_k_weights[i].tolist()
        expert_types = ['Token' if e < 4 else 'Channel' for e in experts]
        print(f'    Sample {i}: Expert {experts[0]} ({expert_types[0]}, w={weights[0]:.2f}), Expert {experts[1]} ({expert_types[1]}, w={weights[1]:.2f})')

    print('\n' + '='*70)
    print('All tests passed!')
    print('='*70)


if __name__ == '__main__':
    main()
