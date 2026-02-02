import re
import matplotlib.pyplot as plt
import os

def parse_log(file_path):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    # Regex for parsing the log lines
    # New format: Epoch [125/200] LR=0.000323 | Train Loss=0.6253, Acc@1=XX.XX% | Val: Loss=0.7654, Acc@1=89.09%
    # Old format: Epoch [125/200] LR=0.000323 | Train Loss=0.6253 | Val: Loss=0.7654, Acc@1=89.09%
    pattern = re.compile(r"Epoch \[(\d+)/\d+\].*Train Loss=([\d.]+)(?:, Acc@1=([\d.]+)%)?.*Val: Loss=([\d.]+), Acc@1=([\d.]+)%")
    
    data = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return [], [], [], [], []

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                tl = float(match.group(2))
                ta = float(match.group(3)) if match.group(3) else 0.0
                vl = float(match.group(4))
                va = float(match.group(5))
                data[epoch] = (tl, ta, vl, va)
    
    sorted_epochs = sorted(data.keys())
    for e in sorted_epochs:
        epochs.append(e)
        train_loss.append(data[e][0])
        train_acc.append(data[e][1])
        val_loss.append(data[e][2])
        val_acc.append(data[e][3])
        
    return epochs, train_loss, train_acc, val_loss, val_acc

def plot_dataset_comparison(results_dir, dataset, output_path):
    # Find all model folders for this dataset
    dataset_dir = os.path.join(results_dir, dataset)
    if not os.path.exists(dataset_dir):
        # Fallback to checking subfolders if structure is results/cls/dataset/model
        dataset_dir = os.path.join(results_dir, 'cifar' if 'cifar' in dataset else dataset)
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            return

    model_logs = {}
    for model_folder in os.listdir(dataset_dir):
        log_path = os.path.join(dataset_dir, model_folder, "log.txt")
        if os.path.exists(log_path):
            model_logs[model_folder] = parse_log(log_path)

    if not model_logs:
        print(f"No log files found for dataset {dataset}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_logs)))
    
    for i, (model_name, data) in enumerate(model_logs.items()):
        epochs, t_loss, t_acc, v_loss, v_acc = data
        color = colors[i]
        
        # Plot Accuracy (Val solid, Train dashed)
        v_mask = [a > 0 for a in v_acc]
        axes[0].plot(np.array(epochs)[v_mask], np.array(v_acc)[v_mask], '-', color=color, label=f'{model_name} (Val)')
        
        t_mask = [a > 0 for a in t_acc]
        if any(t_mask):
            axes[0].plot(np.array(epochs)[t_mask], np.array(t_acc)[t_mask], '--', color=color, alpha=0.4, label=f'{model_name} (Train)')
            
        # Plot Val Loss
        axes[1].plot(epochs, v_loss, '-', color=color, label=model_name)
        axes[1].plot(epochs, t_loss, '--', color=color, alpha=0.3)

    axes[0].set_title(f'Accuracy Comparison - {dataset.upper()}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(fontsize='small', ncol=2)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].set_title(f'Loss Comparison - {dataset.upper()}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize='small')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Dataset plot saved to {output_path}")

if __name__ == "__main__":
    import numpy as np
    results_dir = "results/cls"
    # Create plots for each dataset found in results
    if os.path.exists(results_dir):
        for ds in os.listdir(results_dir):
            plot_dataset_comparison(results_dir, ds, f"comparison_{ds}.png")

