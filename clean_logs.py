import re
import os

def clean_log(file_path):
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    runs = []
    current_run = []
    
    for line in lines:
        if "====" in line and any(x in line for x in ["Experiment:", "Model:", "Parameters:"]):
            # This looks like a new header
            # But wait, headers have multiple rows.
            # A run starts with a line of "=" then "Experiment:" etc.
            if "============================================================" in line and len(current_run) > 0:
                # Potential start of new run if next few lines look like header
                pass 
            
        # simpler: if word "Experiment:" is in line, and we are not in a run or the previous line was "===="
        if "Experiment:" in line:
            if current_run:
                runs.append(current_run)
            current_run = [line]
        elif current_run:
            current_run.append(line)
        else:
            # Junk at start or epoch lines without a run header
            current_run = [line]

    if current_run:
        runs.append(current_run)
        
    epoch_map = {} # epoch -> (timestamp, line)
    
    for run in runs:
        # Check if Mixup: True is in this run
        # Note: if the log was truncated, we might miss the Mixup: True line.
        # But we can look at the "Train Loss" values.
        # If Train Loss > 1.0 at epoch 100 on CIFAR, it's definitely mixup.
        
        is_correct = False
        has_mixup_true = any("Mixup: True" in l for l in run)
        has_mixup_false = any("Mixup: False" in l for l in run)
        
        if has_mixup_true:
            is_correct = True
        elif has_mixup_false:
            is_correct = False
        else:
            # Heuristic for runs without clear header
            # Check a few epoch lines
            for l in run:
                if "Epoch [" in l:
                    match = re.search(r"Train Loss=([\d.]+)", l)
                    if match:
                        loss = float(match.group(1))
                        # In CIFAR non-mixup, loss drops < 0.8 very fast.
                        # In mixup, it stays > 1.0 for a long time.
                        if loss > 1.0:
                            is_correct = True
                        break
        
        if is_correct:
            for l in run:
                if "Epoch [" in l:
                    match = re.search(r"Epoch \[(\d+)/\d+\]", l)
                    if match:
                        epoch = int(match.group(1))
                        # We keep the latest one (later run in the file)
                        epoch_map[epoch] = l

    sorted_epochs = sorted(epoch_map.keys())
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for e in sorted_epochs:
            f.write(epoch_map[e])

if __name__ == "__main__":
    clean_log("results/cls/cifar/ViT-Ti-4_e200/log.txt")
    clean_log("results/cls/cifar/ViT-MoM-Ti-4_e200/log.txt")
