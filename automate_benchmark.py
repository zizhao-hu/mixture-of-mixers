import subprocess
import os
import time
import sys

# Configuration
PYTHON_EXE = r"C:\Users\zizha\anaconda3\python.exe"
RESULTS_DIR = "results/cls"

PLAN = [
    {'dataset': 'cifar10', 'models': ['ViT-Ti/4', 'ViT-MoM-Ti/4', 'ViT-LMoM-Ti/4']},
    {'dataset': 'cifar100', 'models': ['ViT-Ti/4', 'ViT-MoM-Ti/4', 'ViT-LMoM-Ti/4']},
]

def is_running(model_name, dataset):
    try:
        cmd = f'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | Select-Object CommandLine | Where-Object {{ $_.CommandLine -like "*--model {model_name}*" -and $_.CommandLine -like "*--dataset {dataset}*" }}'
        result = subprocess.check_output(['powershell', '-Command', cmd], universal_newlines=True)
        return len(result.strip()) > 0
    except:
        return False

def run_cmd(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}")
    return subprocess.Popen(cmd_list)

def main():
    print("Starting Automated Benchmark Pipeline...")
    
    for stage in PLAN:
        dataset = stage['dataset']
        for model in stage['models']:
            pids = get_pids(model, dataset)
            if pids:
                print(f"Attaching to existing process for {model} on {dataset} (PIDs: {pids})...")
                for pid in pids:
                    try:
                        # On Windows, we can use a simple wait loop or psutil
                        import psutil
                        proc = psutil.Process(pid)
                        proc.wait()
                    except ImportError:
                        # Fallback if psutil not installed
                        while is_running_pid(pid):
                            time.sleep(60)
                print(f"{model} on {dataset} finished.")
            else:
                # Check if already done (results.json exists)
                model_folder = model.replace('/', '-') + "_e200"
                ds_folder = 'cifar' if dataset.startswith('cifar') else dataset
                res_file = os.path.join(RESULTS_DIR, ds_folder, model_folder, "results.json")
                
                if os.path.exists(res_file):
                    print(f"Skipping {model} on {dataset} - already completed.")
                else:
                    # Start training
                    cmd = [
                        PYTHON_EXE, "train_cls.py",
                        "--dataset", dataset,
                        "--model", model,
                        "--epochs", "200",
                        "--mixup",
                        "--results-dir", RESULTS_DIR,
                        "--save-every", "50"
                    ]
                    
                    if model == "ViT-Ti/4" and dataset == "cifar10":
                         best_pth = os.path.join(RESULTS_DIR, "cifar", "ViT-Ti-4_e200", "checkpoints", "best.pt")
                         if os.path.exists(best_pth):
                             cmd.extend(["--resume", best_pth])

                    proc = run_cmd(cmd)
                    print(f"Started {model} on {dataset} (PID: {proc.pid})")
                    proc.wait()
            
            # After each model, update plots and report
            print("Updating plots and reports...")
            try:
                subprocess.run([PYTHON_EXE, "plot_results.py"])
                subprocess.run([PYTHON_EXE, "analyze_model.py"])
            except Exception as e:
                print(f"Post-processing failed: {e}")

def get_pids(model_name, dataset):
    try:
        cmd = f'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | Select-Object ProcessId, CommandLine | Where-Object {{ $_.CommandLine -like "*--model {model_name}*" -and $_.CommandLine -like "*--dataset {dataset}*" }} | Select-Object -ExpandProperty ProcessId'
        result = subprocess.check_output(['powershell', '-Command', cmd], universal_newlines=True)
        return [int(p.strip()) for p in result.splitlines() if p.strip()]
    except:
        return []

def is_running_pid(pid):
    try:
        subprocess.check_output(['powershell', '-Command', f'Get-Process -Id {pid}'], stderr=subprocess.STDOUT)
        return True
    except:
        return False

    print("Pipeline Finished!")

if __name__ == "__main__":
    main()
