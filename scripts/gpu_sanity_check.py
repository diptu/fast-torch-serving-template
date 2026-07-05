"""
gpu_sanity_check.py
====================
Run this after connecting your local editor to the Colab GPU.
It just confirms the kernel actually has a working GPU behind it.

How to run:
  - Zero-touch: `python scripts/colab/run_remote.py scripts/colab/gpu_sanity_check.py`
  - In VS Code: open this file, pick the Colab kernel, click ▶
  - In Jupyter: new notebook, paste this in, Shift+Enter
  - In terminal: jupyter console --existing <url-with-token>
                 then paste each block
"""

import sys

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "❌  No GPU detected. Did you actually pick a GPU runtime on Colab?"
    )

device = torch.device("cuda")
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"GPU:    {name}")
print(f"VRAM:   {props.total_memory / 1e9:.1f} GB")
print(f"CUDA:   {torch.version.cuda}")

# Tiny GPU op to prove compute actually happens on the GPU
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)
c = a @ b
torch.cuda.synchronize()

print(f"\n✅  GPU compute works. Sample output: c[0,0] = {c[0, 0].item():.4f}")
print("You're all set — start loading your real model.")
