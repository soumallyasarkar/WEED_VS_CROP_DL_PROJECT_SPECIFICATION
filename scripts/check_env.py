#!/usr/bin/env python3
"""Environment and dependency verification script."""

import os
import platform
import shutil
import sys
import psutil

def main() -> None:
    print("=" * 60)
    print("ENVIRONMENT CHECK: Weed vs Crop Discrimination Project")
    print("=" * 60)

    # 1. System specs
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.machine()} / {platform.processor()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {platform.python_version()}")
    print(f"CPU Physical Cores: {psutil.cpu_count(logical=False)}, Logical Cores: {psutil.cpu_count(logical=True)}")
    ram = psutil.virtual_memory()
    print(f"RAM: Total={ram.total / (1024**3):.2f} GB, Available={ram.available / (1024**3):.2f} GB")

    # Disk space
    cwd = os.getcwd()
    total, used, free = shutil.disk_usage(cwd)
    print(f"Disk at '{cwd}': Total={total / (1024**3):.2f} GB, Free={free / (1024**3):.2f} GB")

    # 2. PyTorch and GPU
    print("-" * 60)
    import torch
    import torchvision

    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA Device Count: {device_count}")
        print(f"CUDA Device 0: {device_name} ({vram_gb:.2f} GB VRAM)")
        device = "cuda:0"
    else:
        print("Running on CPU only.")
        device = "cpu"
    print(f"Selected Device: {device}")

    # 3. Key dependencies
    print("-" * 60)
    packages = [
        "numpy",
        "pandas",
        "PIL",
        "cv2",
        "sklearn",
        "matplotlib",
        "seaborn",
        "yaml",
        "torchmetrics",
        "pycocotools",
        "streamlit",
        "pytest",
        "tqdm",
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed (no __version__)")
            print(f"  {pkg:15s}: {version}")
        except ImportError as e:
            print(f"  {pkg:15s}: FAILED ({e})")

    print("=" * 60)
    print("Environment check completed successfully.")

if __name__ == "__main__":
    main()
