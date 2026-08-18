"""Verify the local Python ML development environment for DocGuard AI."""

from __future__ import annotations

import platform
import sys


def _cpu_info() -> str:
    """Return a human-readable CPU description when available."""
    processor = platform.processor()
    if processor:
        return processor

    try:
        import os

        if hasattr(os, "sched_getaffinity"):
            cores = len(os.sched_getaffinity(0))
            return f"{platform.machine()} ({cores} logical cores)"
    except (AttributeError, NotImplementedError, OSError):
        pass

    return platform.machine() or "unknown"


def _torch_status() -> tuple[bool, bool | None, str | None]:
    """Return (installed, cuda_available, cuda_version)."""
    try:
        import torch
    except ImportError:
        return False, None, None

    cuda_available = torch.cuda.is_available()
    cuda_version: str | None = None
    if cuda_available:
        cuda_version = torch.version.cuda

    return True, cuda_available, cuda_version


def main() -> int:
    print("DocGuard AI - environment check")
    print("=" * 40)
    print(f"Python version : {sys.version.split()[0]} ({sys.executable})")
    print(f"Operating system : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU information  : {_cpu_info()}")

    torch_installed, cuda_available, cuda_version = _torch_status()
    print(f"PyTorch installed: {torch_installed}")

    if torch_installed:
        print(f"CUDA available   : {cuda_available}")
        if cuda_available and cuda_version:
            print(f"CUDA version     : {cuda_version}")
        elif cuda_available:
            print("CUDA version     : available (version not reported)")
        else:
            print("CUDA version     : n/a (CUDA not available)")
    else:
        print("CUDA available   : n/a (PyTorch not installed)")
        print("CUDA version     : n/a (PyTorch not installed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
