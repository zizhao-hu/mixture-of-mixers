import re
import os

def clean_log_to_epoch(file_path, max_epoch):
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for l in lines:
        if "Epoch [" in l:
            match = re.search(r"Epoch \[(\d+)/\d+\]", l)
            if match:
                epoch = int(match.group(1))
                if epoch <= max_epoch:
                    new_lines.append(l)
        else:
            # Keep headers for now, but usually they are at the top
            new_lines.append(l)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    # ViT: Clean to 50 (last good checkpoint)
    clean_log_to_epoch("results/cls/cifar/ViT-Ti-4_e200/log.txt", 50)
    # MoM: Clean to 0 (restart)
    with open("results/cls/cifar/ViT-MoM-Ti-4_e200/log.txt", "w") as f:
        pass
