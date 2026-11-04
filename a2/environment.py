"""Hardware/software metadata and timing definitions shared by all questions."""
from __future__ import annotations

import platform
from pathlib import Path

import torch

TIMING_NOTES = {
    "train_seconds": "All epochs: DataLoader iteration (incl. CPU transforms), host->device copies "
                     "(no-op on CPU), forward/backward/step. Excludes dataset loading and model "
                     "construction. Device synchronized at start and end (no-op on CPU).",
    "inference_seconds": "Full pass over the test set: DataLoader iteration, host->device copies, "
                         "forward, argmax. Excludes copying predictions to CPU and accuracy "
                         "computation. Device synchronized at start and end (no-op on CPU).",
}


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> dict:
    """GPU fields are null when no GPU is visible (e.g. CPU-only runs)."""
    gpu_visible = torch.cuda.is_available()
    return {
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if gpu_visible else None,
        "gpu_visible_to_pytorch": gpu_visible,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "platform": platform.platform(),
    }
