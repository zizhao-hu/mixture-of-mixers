#!/usr/bin/env python
# Benchmark script for ViT vs ViT-MoM comparison
#
# Runs experiments on CIFAR-10 and CIFAR-100 with multiple model sizes.
#
# Usage:
#   python run_benchmarks.py --dataset cifar10 --quick   # Quick test (10 epochs)
#   python run_benchmarks.py --dataset cifar10           # Full benchmark (200 epochs)
#   python run_benchmarks.py --all                       # Run all benchmarks

import subprocess
import argparse
import os
import json
from datetime import datetime


# Benchmark configurations
BENCHMARKS = {
    'cifar10': {
        'dataset': 'cifar10',
        'epochs': 200,
        'batch_size': 128,
        'lr': 1e-3,
        'models': ['ViT-Ti/4', 'ViT-MoM-Ti/4', 'ViT-S/4', 'ViT-MoM-S/4'],
    },
    'cifar100': {
        'dataset': 'cifar100',
        'epochs': 200,
        'batch_size': 128,
        'lr': 1e-3,
        'models': ['ViT-Ti/4', 'ViT-MoM-Ti/4', 'ViT-S/4', 'ViT-MoM-S/4'],
    },
}


def run_experiment(dataset, model, epochs, batch_size, lr, results_dir, quick=False):
    """Run a single experiment."""
    if quick:
        epochs = 10
    
    cmd = [
        'python', 'train_cls.py',
        '--dataset', dataset,
        '--model', model,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--results-dir', results_dir,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def collect_results(results_dir):
    """Collect results from all experiments (recursive)."""
    results = {}
    
    for root, dirs, files in os.walk(results_dir):
        if 'results.json' in files:
            results_file = os.path.join(root, 'results.json')
            exp_name = os.path.relpath(root, results_dir).replace(os.sep, '_')
            
            with open(results_file, 'r') as f:
                data = json.load(f)
                results[exp_name] = {
                    'model': data['args']['model'],
                    'dataset': data['args']['dataset'],
                    'best_acc1': data['best_acc1'],
                    'final_acc1': data['final_acc1'],
                    'epochs': data['args']['epochs'],
                }
    
    return results


def print_summary(results):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Group by dataset
    datasets = {}
    for exp_name, data in results.items():
        ds = data['dataset']
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(data)
    
    for ds, exps in datasets.items():
        print(f"\n{ds.upper()}")
        print("-"*60)
        print(f"{'Model':<20} {'Best Acc@1':>12} {'Final Acc@1':>12} {'Epochs':>8}")
        print("-"*60)
        
        # Sort by model name
        exps.sort(key=lambda x: x['model'])
        
        for exp in exps:
            print(f"{exp['model']:<20} {exp['best_acc1']:>11.2f}% {exp['final_acc1']:>11.2f}% {exp['epochs']:>8}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Run ViT vs ViT-MoM benchmarks")
    parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100', 'all'], 
                        default='cifar10', help="Dataset to benchmark")
    parser.add_argument("--model", type=str, default=None, 
                        help="Specific model to run (overrides default list)")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 epochs")
    parser.add_argument("--results-dir", type=str, default="./results/cls", 
                        help="Results directory")
    parser.add_argument("--collect-only", action="store_true", 
                        help="Only collect and print existing results")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    
    args = parser.parse_args()
    
    # Collect only mode
    if args.collect_only:
        results = collect_results(args.results_dir)
        print_summary(results)
        return
    
    # Determine which benchmarks to run
    if args.all:
        benchmark_names = list(BENCHMARKS.keys())
    else:
        benchmark_names = [args.dataset]
    
    # Run benchmarks
    successful = []
    failed = []
    
    for bench_name in benchmark_names:
        bench = BENCHMARKS[bench_name]
        models = [args.model] if args.model else bench['models']
        
        for model in models:
            success = run_experiment(
                dataset=bench['dataset'],
                model=model,
                epochs=bench['epochs'],
                batch_size=bench['batch_size'],
                lr=bench['lr'],
                results_dir=args.results_dir,
                quick=args.quick,
            )
            
            if success:
                successful.append(f"{model} on {bench['dataset']}")
            else:
                failed.append(f"{model} on {bench['dataset']}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)}")
    for s in successful:
        print(f"  ✓ {s}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  ✗ {f}")
    
    # Collect and print results
    results = collect_results(args.results_dir)
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
