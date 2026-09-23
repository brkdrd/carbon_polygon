#!/usr/bin/env python3
"""Preflight: does the GPU actually reach this container, and do kernels RUN?

Five seconds, so a broken passthrough surfaces before the hours-long stages
rather than after them.

`torch.cuda.is_available()` is not sufficient on Blackwell (RTX 50xx, sm_120):
a torch built only up to sm_90 still enumerates the device and reports True,
then dies on the first kernel with "no kernel image is available". So this
launches a real matmul and synchronises on it.

Exit 0 = the GPU stages will work. Exit 1 = they will not, with the reason.
"""
from __future__ import annotations

import sys

import torch

print(f"torch {torch.__version__} | built against CUDA {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\nFAIL: torch.cuda.is_available() is False — no GPU reached this "
          "container.\n"
          "  * Docker Desktop: Settings -> Resources -> WSL Integration must be\n"
          "    on for the distro you run from; GPU support is then built in.\n"
          "  * The NVIDIA driver must be installed on WINDOWS, not inside WSL —\n"
          "    installing a Linux driver in the distro breaks the projection.\n"
          "  * Independent check, outside this project:\n"
          "      docker run --rm --gpus all "
          "nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  device {i}: {p.name} | sm_{p.major}{p.minor} | "
          f"{p.total_memory / 1e9:.1f} GB")

arch = torch.cuda.get_arch_list()
print(f"  this torch ships kernels for: {' '.join(arch)}")

try:
    a = torch.randn(1024, 1024, device="cuda")
    v = float((a @ a).sum())
    torch.cuda.synchronize()
    print(f"  kernel launch: OK (matmul checksum {v:.3e})")
except Exception as e:                                   # noqa: BLE001
    print(f"\nFAIL: the device enumerates but kernels do not run:\n  {e}\n"
          "  That is the sm_120 symptom: this torch has no kernels for the\n"
          "  card and no PTX to JIT from. The sonata image pins torch 2.7 +\n"
          "  CUDA 12.8 precisely to cover Blackwell — if you see this, the\n"
          "  image was built from a different base than the repo specifies.")
    sys.exit(1)

cap = torch.cuda.get_device_capability(0)
want = f"sm_{cap[0]}{cap[1]}"
if want not in arch:
    print(f"\nNOTE: {want} is not in the built arch list above, so kernels are\n"
          f"  being JIT-compiled from PTX. That works — it is how spconv-cu126\n"
          f"  covers this card — but the first kernel of each kind is slow.\n"
          f"  Expect a stall early in gw_train, then normal speed.")

print("\nGPU preflight passed — gw_train / gw_infer will have the card.")
